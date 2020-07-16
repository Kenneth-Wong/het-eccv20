import os
import os.path as osp
import json
import h5py
import numpy as np
from data.vg200.utils.config import *
import pickle
import torch


def load_vg_iminfos():
    """
    :return: a list containing im tuple (path, vg_id, and coco_id)
    """
    image_meta = json.load(open(meta_file))
    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    image_infos = []
    for i in image_meta:
        basename = str(i['image_id']) + '.jpg'
        if basename in corrupted_ims:
            continue
        if osp.isfile(osp.join(IMAGE_DIR1, basename)):
            im_path = osp.join(IMAGE_DIR1, basename)
        else:
            im_path = osp.join(IMAGE_DIR2, basename)
        image_infos.append((im_path, i['image_id'], i['coco_id']))

    assert len(image_infos) == 108073
    return image_infos

def vecDist(vec1, vec2):
    v1sum = torch.sum(vec1**2, 1, keepdim=True)
    v2sum = torch.transpose(torch.sum(vec2**2, 1, keepdim=True), 0, 1)
    product = torch.matmul(vec1, torch.transpose(vec2, 0, 1))
    dist = v1sum + v2sum - 2 * product
    return dist

def db2vg():
    """transfer corrupted-not-removed db indices (108077) to vg indexes"""
    dbidx2vgidx = {}
    image_meta = json.load(open(meta_file))
    for i, item in enumerate(image_meta):
        dbidx2vgidx[i] = item['image_id']
    vgidx2dbidx = {dbidx2vgidx[i]: i for i in dbidx2vgidx}
    return dbidx2vgidx, vgidx2dbidx

def cleaneddb2vg():
    """transfer corrupted-removed db indices (108073) to vg indexes"""
    dbidx2vgidx = {}
    image_meta = json.load(open(meta_file))
    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    keep = []
    for i, item in enumerate(image_meta):
        basename = str(item['image_id']) + '.jpg'
        if basename in corrupted_ims:
            continue
        keep.append(item)
    for i, item in enumerate(keep):
        dbidx2vgidx[i] = item['image_id']
    vgidx2dbidx = {dbidx2vgidx[i]: i for i in dbidx2vgidx}
    return dbidx2vgidx, vgidx2dbidx


def load_vghdf5():
    info = json.load(open(dict_file))
    class_to_ind = info['label_to_idx']
    class_to_ind['__background__'] = 0
    classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])

    predicate_to_ind = info['predicate_to_idx']
    predicate_to_ind['__background__'] = 0
    predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    roi_h5 = h5py.File(roidb_file, 'r')
    im_h5 = h5py.File(imdb_file, 'r')
    im_refs = im_h5['images']
    im_sizes = np.vstack([im_h5['image_widths'], im_h5['image_heights']]).transpose()

    im_to_first_box = roi_h5['img_to_first_box']
    im_to_last_box = roi_h5['img_to_last_box']
    all_boxes = roi_h5['boxes_%i' % 1024][:]  # will index later
    assert (np.all(all_boxes[:, :2] >= 0))  # sanity check
    assert (np.all(all_boxes[:, 2:] > 0))  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - np.floor(all_boxes[:, 2:] / 2)
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:] - 1
    labels = roi_h5['labels'][:, 0]

    im_to_first_rel = roi_h5['img_to_first_rel']
    im_to_last_rel = roi_h5['img_to_last_rel']
    relations = roi_h5['relationships'][:]
    relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (relations.shape[0] == relation_predicates.shape[0])
    return classes, predicates, im_refs, im_sizes, all_boxes, im_to_first_box, im_to_last_box, \
           im_to_first_rel, im_to_last_rel, labels, relations, relation_predicates


def get_image_db(classes, predicates, im_infos, im_sizes, all_boxes, im_to_first_box, im_to_last_box,
                 im_to_first_rel, im_to_last_rel, labels, relations, relation_predicates):
    if osp.isfile(osp.join(CACHE_DIR, 'vghdf5_db.pkl')):
        with open(osp.join(CACHE_DIR, 'vghdf5_db.pkl'), 'rb') as f:
            db = pickle.load(f)
        return db
    db = {}
    db['classes'] = classes
    db['predicates'] = predicates
    imdb = []
    for db_idx in range(len(im_infos)):
    #for db_idx in range(2):
        w, h = im_sizes[db_idx, :]
        im_path, vg_id, coco_id = im_infos[db_idx]

        boxes = all_boxes[im_to_first_box[db_idx]:im_to_last_box[db_idx] + 1, :]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        gt_classes = labels[im_to_first_box[db_idx]:im_to_last_box[db_idx] + 1]

        gt_relations = []
        if im_to_first_rel[db_idx] >= 0:  # if image has relations
            this_predicates = relation_predicates[im_to_first_rel[db_idx]:im_to_last_rel[db_idx] + 1]
            obj_idx = relations[im_to_first_rel[db_idx]:im_to_last_rel[db_idx] + 1]
            obj_idx = obj_idx - im_to_first_box[db_idx]
            assert (np.all(obj_idx >= 0) and np.all(obj_idx < boxes.shape[0]))  # sanity check
            for j, p in enumerate(this_predicates):
                gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])

        gt_relations = np.array(gt_relations)

        imdb.append({'w': w, 'h': h, 'im_path':im_path, 'boxes': boxes, 'gt_classes': gt_classes,
                     'gt_relations': gt_relations, 'vg_id': vg_id, 'coco_id': coco_id, 'areas': areas})
    db['imdb'] = imdb

    with open(osp.join(CACHE_DIR, 'vghdf5_db.pkl'), 'wb') as f:
        pickle.dump(db, f)
    return db


def get_hierarchical_rels(db, geo_thresh=0.6, pos_thresh=0.985):
    # extract all relations
    imdb = db['imdb']
    classes = db['classes']
    predicates = db['predicates']
    geo_overlap_comp_small = {}
    pos_overlap_comp_small = {}
    for i in range(len(imdb)):
        gt_relations = imdb[i]['gt_relations']
        gt_classes = imdb[i]['gt_classes']
        boxes = imdb[i]['boxes']
        for rel in gt_relations:
            triplet = (classes[gt_classes[rel[0]]], predicates[rel[2]], classes[gt_classes[rel[1]]])
            if predicates[rel[2]] in geo:
                sub_box, obj_box = boxes[rel[0]], boxes[rel[1]]
                x11, y11, x12, y12 = sub_box
                x21, y21, x22, y22 = obj_box

                iw = np.minimum(x12, x22) - np.maximum(x11, x21) + 1
                ih = np.minimum(y12, y22) - np.maximum(y11, y21) + 1
                if iw > 1 and ih > 1:
                    sub_area = (x12 - x11 + 1) * (y12 - y11 + 1)
                    obj_area = (x22 - x21 + 1) * (y22 - y21 + 1)
                    if triplet in geo_overlap_comp_small:
                        geo_overlap_comp_small[triplet].append((iw * ih * 1.0) / min(sub_area, obj_area))
                    else:
                        geo_overlap_comp_small[triplet] = [(iw * ih * 1.0) / min(sub_area, obj_area)]
            elif predicates[rel[2]] in pos:
                sub_box, obj_box = boxes[rel[0]], boxes[rel[1]]
                x11, y11, x12, y12 = sub_box
                x21, y21, x22, y22 = obj_box

                iw = np.minimum(x12, x22) - np.maximum(x11, x21) + 1
                ih = np.minimum(y12, y22) - np.maximum(y11, y21) + 1
                if iw > 1 and ih > 1:
                    sub_area = (x12 - x11 + 1) * (y12 - y11 + 1)
                    obj_area = (x22 - x21 + 1) * (y22 - y21 + 1)
                    if triplet in pos_overlap_comp_small:
                        pos_overlap_comp_small[triplet].append((iw * ih * 1.0) / min(sub_area, obj_area))
                    else:
                        pos_overlap_comp_small[triplet] = [(iw * ih * 1.0) / min(sub_area, obj_area)]

    selected_geo_triplets = []
    selected_pos_triplets = []
    for triplet in geo_overlap_comp_small:
        m = np.mean(geo_overlap_comp_small[triplet])
        if m < geo_thresh:
            selected_geo_triplets.append(triplet)
    for triplet in pos_overlap_comp_small:
        m = np.mean(pos_overlap_comp_small[triplet])
        if m > pos_thresh:
            selected_pos_triplets.append(triplet)

    sel_geo_sample_num, sel_pos_sample_num = 0, 0
    both_image_num = 0
    single_image_num = 0

    ## annotate imdb
    for i in range(len(imdb)):
        hierar_rels = []
        coord_rels = []
        gt_relations = imdb[i]['gt_relations']
        gt_classes = imdb[i]['gt_classes']
        for j, rel in enumerate(gt_relations):
            triplet = (classes[gt_classes[rel[0]]], predicates[rel[2]], classes[gt_classes[rel[1]]])
            if triplet in selected_pos_triplets:
                hierar_rels.append(j)
            elif triplet in selected_geo_triplets:
                coord_rels.append(j)
        imdb[i]['hierar_relations'] = hierar_rels
        imdb[i]['coord_relations'] = coord_rels

        sel_geo_sample_num += len(coord_rels)
        sel_pos_sample_num += len(hierar_rels)
        both_image_num = both_image_num + 1 if len(hierar_rels) > 0 and len(coord_rels) > 0 else both_image_num
        single_image_num = single_image_num + 1 if len(hierar_rels) > 0 or len(coord_rels) > 0 else single_image_num

    print('geo_triplets num: %d \t geo_triplets sample num: %d \t pos_triplets num: %d \t pos_triplets sample num: %d \t '
          'both image num: %d \t both_rel/image: %f, \t single image num: %d \t single_rel/image: %f' %
          (len(selected_geo_triplets), sel_geo_sample_num,
                                                          len(selected_pos_triplets), sel_pos_sample_num,
                                                          both_image_num, (sel_geo_sample_num + sel_pos_sample_num) * 1.0 / both_image_num,
           single_image_num, (sel_geo_sample_num + sel_pos_sample_num) * 1.0 / single_image_num))

    with open(osp.join(CACHE_DIR, 'vghdf5_imdb_addRel.pkl'), 'wb') as f:
        pickle.dump(imdb, f)

    return imdb


def decide_tree_level(trees):
    """
    :param trees: dict, key is parent, value is a son list. root at -1
    :return: a dict that decide the level of each node, root at level 0
    """
    basic_level = 0
    level_dict = {}
    def traverse_tree(node_id, level):
        level_dict[node_id] = level
        if node_id not in trees:
            return
        son_list = trees[node_id]
        for son in son_list:
            traverse_tree(son, level+1)

    traverse_tree(-1, basic_level)
    return level_dict

def get_hierarchical_trees(db, pos_thresh=0.9):
    if osp.isfile(osp.join(CACHE_DIR, 'vghdf5_db_trees.pkl')):
        with open(osp.join(CACHE_DIR, 'vghdf5_db_trees.pkl'), 'rb') as f:
            db = pickle.load(f)
        return db

    # extract all relations
    imdb = db['imdb']
    classes = db['classes']
    predicates = db['predicates']
    for i in range(len(imdb)):
        gt_relations = imdb[i]['gt_relations']
        gt_classes = imdb[i]['gt_classes']
        boxes = imdb[i]['boxes']
        areas = imdb[i]['areas']
        parents = (np.ones(boxes.shape[0]) * -1).astype(np.int32)
        recorded_pairs = {}
        for rel in gt_relations:
            rel_name = predicates[rel[2]]
            if rel_name in pos or rel_name == 'on':  # possesive relationships
                b1, b2 = boxes[rel[0]], boxes[rel[1]]
                x11, y11, x12, y12 = b1
                x21, y21, x22, y22 = b2
                # set p_mark: 1 for the first is parent, 2 for the second is parent
                p_mark = 0
                iw = np.minimum(x12, x22) - np.maximum(x11, x21) + 1
                ih = np.minimum(y12, y22) - np.maximum(y11, y21) + 1
                if iw > 1 and ih > 1: # cond1: overlap
                    area1, area2 = areas[rel[0]], areas[rel[1]]
                    ratio = (iw * ih * 1.0) / min(area1, area2)
                    if ratio > pos_thresh: # cond2: overlap/smaller area must be bigger than pos_thresh
                        # decide which is the real subject, and the real object
                        # generally, the one with larger area is subject
                        if area1 > area2: # b1 is subject, b2 is object
                            p_mark = 1
                        elif area1 < area2:
                            p_mark = 2
                        else: # same, experimentally, consider the relation
                            if rel_name in ['has', 'wearing', 'wears']:
                                p_mark = 1
                            elif rel_name in ['belonging to', 'of', 'part of', 'to']:
                                p_mark = 2

                        # do not consider the same pair again
                        pair = (rel[0], rel[1])
                        if (rel[0], rel[1]) in recorded_pairs or (rel[1], rel[0]) in recorded_pairs:
                            continue

                        # now check whether the new added hierarchical relation is safe
                        if p_mark == 1: ## the first one is parent, check the parent of the second
                            if parents[rel[1]] == -1: # safe
                                parents[rel[1]] = rel[0]
                                recorded_pairs[pair] = ratio
                            else:  # consider whether to change the parent
                                # read the old ratio from recorded_pairs
                                if (rel[1], parents[rel[1]]) in recorded_pairs:
                                    old_ratio = recorded_pairs[(rel[1], parents[rel[1]])]
                                else:
                                    old_ratio = recorded_pairs[(parents[rel[1]], rel[1])]
                                if old_ratio < ratio: # change parent
                                    # delete old item in recorded_pairs
                                    if (rel[1], parents[rel[1]]) in recorded_pairs:
                                        recorded_pairs.pop((rel[1], parents[rel[1]]))
                                    else:
                                        recorded_pairs.pop((parents[rel[1]], rel[1]))
                                    recorded_pairs[pair] = ratio
                                    parents[rel[1]] = rel[0]
                        else:  ## the second one is parent, check the parent of the first
                            if parents[rel[0]] == -1: # safe
                                parents[rel[0]] = rel[1]
                                recorded_pairs[pair] = ratio
                            else:  # consider whether to change the parent
                                # read the old ratio from recorded_pairs
                                if (rel[0], parents[rel[0]]) in recorded_pairs:
                                    old_ratio = recorded_pairs[(rel[0], parents[rel[0]])]
                                else:
                                    old_ratio = recorded_pairs[(parents[rel[0]], rel[0])]
                                if old_ratio < ratio: # change parent
                                    # delete old item in recorded_pairs
                                    if (rel[0], parents[rel[0]]) in recorded_pairs:
                                        recorded_pairs.pop((rel[0], parents[rel[0]]))
                                    else:
                                        recorded_pairs.pop((parents[rel[0]], rel[0]))
                                    recorded_pairs[pair] = ratio
                                    parents[rel[0]] = rel[1]
        tree_info = {}
        for k, p in enumerate(parents):
            if p not in tree_info:
                tree_info[p] = [k]
            else:
                tree_info[p].append(k)
        for p in tree_info:
            tree_info[p] = sorted(tree_info[p])

        db['imdb'][i]['trees'] = tree_info

        db['imdb'][i]['levels'] = decide_tree_level(tree_info)


    with open(osp.join(CACHE_DIR, 'vghdf5_db_trees.pkl'), 'wb') as f:
        pickle.dump(db, f)

    return db




