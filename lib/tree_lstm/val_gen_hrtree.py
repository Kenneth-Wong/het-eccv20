from dataloaders.visual_genome200 import VG200, VGDataLoader
#from dataloaders.visual_genome import VGDataLoader, VG
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from lib.word_vectors import obj_edge_vectors
import numpy as np
from config import IM_SCALE
import random

from lib.tree_lstm import tree_utils
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
from lib.tree_lstm.def_tree import ArbitraryTree
from config import LOG_SOFTMAX
from lib.fpn.box_intersections_cpu.bbox import bbox_intersections as bbox_intersections_np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from skimage.transform import resize


def generate_forest(im, im_ind, box_priors, scores, obj_label, pick_parent, isc_thresh, child_order='leftright'):
    """
    generate a list of trees that covers all the objects in a batch
    im_inds: [obj_num]
    box_priors: [obj_num, (x1, y1, x2, y2)]
    pair_scores: [obj_num, obj_num]

    output: list of trees, each present a chunk of overlaping objects
    """
    output_forest = []  # the list of trees, each one is a chunk of overlapping objects
    num_obj = box_priors.shape[0]

    # select the nodes from the same image
    node_container = []
    remain_index = []
    root_idx = num_obj
    root_roi = Variable(torch.randn(1, 4).fill_(0))
    root_roi[0, 0] = 0
    root_roi[0, 1] = 0
    root_roi[0, 2] = IM_SCALE
    root_roi[0, 3] = IM_SCALE
    root = ArbitraryTree(root_idx, -1, -1, root_roi,
                            im_ind, is_root=True)

    # put all nodes into node container
    for idx in range(num_obj):
        if obj_label is not None:
            label = int(obj_label[idx])
        else:
            label = -1
        new_node = ArbitraryTree(idx, scores[idx], label, box_priors[idx], im_ind)
        node_container.append(new_node)
        remain_index.append(int(idx))

    # iteratively generate tree
    gen_hrtree(node_container, root, remain_index, pick_parent=pick_parent, isc_thresh=isc_thresh, child_order=child_order)

    return root


def gen_hrtree(node_container, root, remain_index, pick_parent='isc', isc_thresh=0.9, child_order='leftright'):
    num_nodes = len(node_container)
    if num_nodes == 0:
        return

    # first step: sort the rois according to areas
    sorted_node_container, sorted_remain_index = sorted_areas(node_container, remain_index)
    areas, bbox_intersection = get_all_boxes_info(sorted_node_container)

    sort_key = 1 if pick_parent == 'isc' else 2
    for i in range(num_nodes):
        current_node = sorted_node_container[i]
        possible_parent = []
        for j in range(0, i):  # all nodes that are larger than current_node
            M = bbox_intersection[i, j]
            N = bbox_intersection[j, i]
            if M > isc_thresh:
                possible_parent.append((j, N, areas[j]))
        if len(possible_parent) == 0:
            # assign the parrent of i as root
            root.add_child(current_node)
        else:
            if pick_parent != 'area' and pick_parent != 'isc':
                raise NotImplementedError('%s for pick_parent not implemented' % pick_parent)

            parent_id = sorted(possible_parent, key=lambda d: d[sort_key], reverse=True)[0][0]
            sorted_node_container[parent_id].add_child(current_node)
    # sort the children
    sort_childs(root, child_order)

def sort_childs(root, order='leftright'):
    if len(root.children) == 0:
        return
    children = root.children
    boxes = np.vstack([n.box.cpu().data.numpy() for n in children])
    node_scores = np.array([n.score for n in children])
    if order == 'leftright':
        scores = (boxes[:, 0] + boxes[:, 2]) / 2
        scores = scores / (np.max(scores) + 1)
    elif order == 'size':
        scores = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        scores = scores / (np.max(scores) + 1)
    elif order == 'confidence':
        scores = node_scores
    elif order == 'random':
        scores = np.random.rand(len(children))
    else:
        raise NotImplementedError('Unknown sorting method: %s' % order)
    sorted_id = np.argsort(-scores)
    root.children = [children[i] for i in sorted_id]
    for i in range(len(root.children)):
        sort_childs(root.children[i], order)



def sorted_areas(node_container, remain_index):
    areas = [((n.box[3] - n.box[1] + 1) * (n.box[2] - n.box[0] + 1)).cpu().data.numpy()[0] for n in node_container]
    sorted_id = np.argsort(-1 * np.array(areas))
    new_node_container = [node_container[i] for i in sorted_id]
    new_remain_index = [remain_index[i] for i in sorted_id]
    return new_node_container, new_remain_index


def get_all_boxes_info(node_container):
    areas = np.array([((n.box[3] - n.box[1] + 1) * (n.box[2] - n.box[0] + 1)).cpu().data.numpy()[0] for n in node_container])
    boxes = np.vstack([n.box.cpu().data.numpy() for n in node_container])
    bbox_intersection = np.transpose(
        bbox_intersections_np(boxes, boxes))  # row-wise, box of each row is the denominator
    return areas, bbox_intersection


def draw_tree_region_v2(tree, image, example_id, pred_labels):
    """
    tree: A tree structure
    image: origin image batch [batch_size, 3, IM_SIZE, IM_SIZE]
    output: a image with roi bbox, the color of box correspond to the depth of roi node
    """
    sample_image = image.view(image.shape[1:]).clone()
    sample_image = (revert_normalize(sample_image) * 255).int()
    sample_image = torch.clamp(sample_image, 0, 255)
    sample_image = sample_image.permute(1, 2, 0).contiguous().data.cpu().numpy().astype(dtype=np.uint8)
    sample_image = Image.fromarray(sample_image, 'RGB').convert("RGBA")

    draw = ImageDraw.Draw(sample_image)
    draw_box(draw, tree, pred_labels)

    sample_image.save('./output/example/' + str(example_id) + '_box' + '.png')

    # print('saved img ' + str(example_id))


def draw_box(draw, tree, labels, is_root=True):
    global idx_to_classes
    x1, y1, x2, y2 = int(tree.box[0]), int(tree.box[1]), int(tree.box[2]), int(tree.box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline="red")
    draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill="red")
    if not is_root:
        node_label = int(labels[int(tree.index)])
        draw.text((x1, y1), idx_to_classes[node_label])

    for i in range(len(tree.children)):
        draw_box(draw, tree.children[i], labels, is_root=False)


def draw_tree_region(tree, image, example_id):
    """
    tree: A tree structure
    image: origin image batch [batch_size, 3, IM_SIZE, IM_SIZE]
    output: a image display regions in a tree structure
    """
    sample_image = image.view(image.shape[1:]).clone()
    sample_image = (revert_normalize(sample_image) * 255).int()
    sample_image = torch.clamp(sample_image, 0, 255)
    sample_image = sample_image.permute(1, 2, 0).contiguous().data.cpu().numpy().astype(dtype=np.uint8)


    depth = tree.max_depth()
    width = tree.leafcount()
    tree_img = create_tree_img(depth, width, 64)
    tree_img = write_cell(sample_image, tree_img, (tree_img.shape[1]/2, 0), tree, 64, tree_img.shape[1], tree_img.shape[0])

    im = Image.fromarray(sample_image, 'RGB')
    tree_img = Image.fromarray(tree_img, 'RGB')
    im.save('./output/example/' + str(example_id) + '_origin' + '.jpg')
    tree_img.save('./output/example/' + str(example_id) + '_tree' + '.jpg')

    if example_id % 200 == 0:
        print('saved img ' + str(example_id))


def write_cell(orig_img, tree_img, draw_coord, tree, cell_size, width, height):
    """
    orig_img: original image
    tree_img: draw roi tree
    draw_box: the whole bbox used to draw this sub-tree [x1,y1,x2,y2]
    tree: a sub-tree
    cell_size: size of each roi
    """
    x, y = draw_coord
    if tree is None:
        return tree_img
    # draw
    roi = orig_img[int(tree.box[1]):int(tree.box[3]), int(tree.box[0]):int(tree.box[2]), :]
    roi = Image.fromarray(roi, 'RGB')
    roi = roi.resize((cell_size, cell_size))
    roi = np.array(roi)
    draw_x1 = int(max(x - cell_size / 2, 0))
    draw_x2 = int(min(draw_x1 + cell_size, width - 1))
    draw_y1 = y
    draw_y2 = min(y + cell_size, height - 1)
    tree_img[draw_y1:draw_y2, draw_x1:draw_x2, :] = roi[:draw_y2 - draw_y1, :draw_x2 - draw_x1, :]
    # recursive draw
    children = tree.children
    child_leafcounts = [children[i].leafcount() if children[i].leafcount() else 1 for i in range(len(children))]
    leafpoint = x - sum(child_leafcounts) * cell_size / 2
    cumpoint = 0
    for child, point in zip(tree.children, child_leafcounts):
        centerpoint = leafpoint + cell_size * cumpoint + cell_size * point / 2
        cumpoint += point
        tree_img = write_cell(orig_img, tree_img, (centerpoint, y + cell_size + 1),
                              child, cell_size, width, height)

    return tree_img


def create_tree_img(depth, width, cell_size):
    height = cell_size * (depth + 1)
    width = cell_size * (width + 2)
    return np.zeros((height, width, 3)).astype(dtype=np.uint8)


def revert_normalize(image):
    image[0, :, :] = image[0, :, :] * 0.229
    image[1, :, :] = image[1, :, :] * 0.224
    image[2, :, :] = image[2, :, :] * 0.225

    image[0, :, :] = image[0, :, :] + 0.485
    image[1, :, :] = image[1, :, :] + 0.456
    image[2, :, :] = image[2, :, :] + 0.406

    return image


VGdata = VG200

train, val, test = VGdata.splits(num_val_im=-1, filter_duplicate_rels=True,
                          use_proposals=False,
                          filter_non_overlap=False)

idx_to_classes = train.ind_to_classes
idx_to_predicates = train.ind_to_predicates

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=1,
                                               num_workers=0,
                                               num_gpus=1)

for b, batch in enumerate(train_loader):
    if b >= 200:
        break
    img, im_sizes, _, gt_boxes, gt_classes, rels, proposals, _, sal_map = batch[0]
    tree = generate_forest(img, b, gt_boxes, np.ones(gt_boxes.shape[0], dtype=np.int32), gt_classes[:, 1], 'isc', 0.9)
    draw_tree_region_v2(tree, img, b, gt_classes[:, 1])
    draw_tree_region(tree, img, b)