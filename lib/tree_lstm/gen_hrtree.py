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


def generate_forest(im_inds, box_priors, scores, obj_label, pick_parent, isc_thresh, child_order='leftright',
                    sal_maps=None, depth_maps=None):
    """
    generate a list of trees that covers all the objects in a batch
    im_inds: [obj_num]
    box_priors: [obj_num, (x1, y1, x2, y2)]
    pair_scores: [obj_num, obj_num]

    output: list of trees, each present a chunk of overlaping objects
    """
    output_forest = []  # the list of trees, each one is a chunk of overlapping objects
    num_obj = box_priors.shape[0]

    # make forest
    group_id = 0

    gen_tree_loss_per_batch = []
    entropy_loss = []

    while (torch.nonzero(im_inds == group_id).numel() > 0):
        # select the nodes from the same image
        node_container = []
        remain_index = []
        picked_list = torch.nonzero(im_inds == group_id).view(-1)
        root_idx = picked_list[-1]

        root = ArbitraryTree(root_idx, -1, -1, box_priors[int(root_idx)],
                                im_inds[int(root_idx)], is_root=True)

        # put all nodes into node container
        for idx in picked_list[:-1]:
            if obj_label is not None:
                label = int(obj_label[idx])
            else:
                label = -1
            new_node = ArbitraryTree(idx, scores[idx], label, box_priors[idx], im_inds[idx])
            node_container.append(new_node)
            remain_index.append(int(idx))

        # iteratively generate tree
        gen_hrtree(node_container, root, remain_index, pick_parent=pick_parent, isc_thresh=isc_thresh, child_order=child_order,
                   sal_maps=sal_maps, depth_maps=depth_maps)

        output_forest.append(root)
        group_id += 1

    return output_forest

def get_depths(depth_map, node_container):
    depths = np.zeros(len(node_container))
    depth_map_np = depth_map.squeeze().data.cpu().numpy()
    for i, n in enumerate(node_container):
        box = [int(n.box[0].cpu().data.numpy()[0]), int(n.box[1].cpu().data.numpy()[0]),
               int(n.box[2].cpu().data.numpy()[0]), int(n.box[3].cpu().data.numpy()[0])]
        depth_patch = depth_map_np[box[1]:(box[3] + 1), box[0]:(box[2] + 1)]

        depth_array = depth_patch.reshape(-1)
        length = len(depth_array)
        depth_array = np.sort(depth_array)
        per_len = length // 4
        if length <= 4:
            depths[i] = np.mean(depth_array)
        else:
            depths[i] = np.mean(depth_array[per_len:(length-per_len)])
    return depths

def gen_hrtree(node_container, root, remain_index, pick_parent='area', isc_thresh=0.9, child_order='leftright',
               sal_maps=None, depth_maps=None):
    num_nodes = len(node_container)
    if num_nodes == 0:
        return

    root_area = ((root.box[3] - root.box[1] + 1) * (root.box[2] - root.box[0] + 1)).cpu().data.numpy()[0]
    # first step: sort the rois according to areas
    sorted_node_container, sorted_remain_index = sorted_areas(node_container, remain_index)
    areas, bbox_intersection = get_all_boxes_info(sorted_node_container)
    box_depths = None
    if depth_maps is not None:
        box_depths = get_depths(depth_maps[sorted_node_container[0].im_idx], sorted_node_container)

    if pick_parent == 'isc':
        sort_key = 1
    elif pick_parent == 'area':
        sort_key = 2
    elif pick_parent == 'depth' and box_depths is not None:
        sort_key = 3
    else:
        raise NotImplementedError
    for i in range(num_nodes):
        current_node = sorted_node_container[i]
        possible_parent = []
        for j in range(0, i):  # all nodes that are larger than current_node
            M = bbox_intersection[i, j]
            N = bbox_intersection[j, i]
            if M > isc_thresh:
                if depth_maps is not None:
                    depth_diff = np.abs(box_depths[i] - box_depths[j])
                else:
                    depth_diff = -1
                    assert sort_key != 3
                possible_parent.append((j, N, areas[j], depth_diff))
        if len(possible_parent) == 0:
            # assign the parrent of i as root
            if sal_maps is not None:
                area_feats = current_node.box.new(1, 3).fill_(0)
                area_feats[0, 0] = areas[i]
                area_feats[0, 1] = root_area
                area_feats[0, 2] = areas[i] / root_area
                sal_feats = sal_maps[current_node.index].view(1, -1)
                current_node.sal_feats = torch.cat((sal_feats, area_feats), 1)
            root.add_child(current_node)
        else:
            if pick_parent != 'area' and pick_parent != 'isc' and pick_parent != 'depth':
                raise NotImplementedError('%s for pick_parent not implemented' % pick_parent)

            parent_id = sorted(possible_parent, key=lambda d: d[sort_key], reverse=True)[0][0]
            if sal_maps is not None:
                area_feats = current_node.box.new(1, 3).fill_(0)
                area_feats[0, 0] = areas[i]
                area_feats[0, 1] = areas[parent_id]
                area_feats[0, 2] = areas[i] / areas[parent_id]
                sal_feats = sal_maps[current_node.index].view(1, -1)
                current_node.sal_feats = torch.cat((sal_feats, area_feats), 1)
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
