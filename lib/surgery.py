# create predictions from the other stuff
"""
Go from proposals + scores to relationships.

pred-cls: No bbox regression, obj dist is exactly known
sg-cls : No bbox regression
sg-det : Bbox regression

in all cases we'll return:
boxes, objs, rels, pred_scores

"""

import numpy as np
import torch
from lib.pytorch_misc import unravel_index, enumerate_by_image
from lib.fpn.box_utils import bbox_overlaps
# from ad3 import factor_graph as fg
from time import time
from torch.autograd import Variable


def filter_dets(boxes, obj_scores, obj_classes, rel_inds, pred_scores, gt_boxes, gt_classes, gt_rels,
                rel_rank_scores=None,
                forest=None, return_forest=False, im_inds=None):
    """
    Filters detections....
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores

    """
    if boxes.dim() != 2:
        raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

    num_box = boxes.size(0)
    assert obj_scores.size(0) == num_box

    assert obj_classes.size() == obj_scores.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 2
    assert pred_scores.size(0) == num_rel

    obj_scores0 = obj_scores.data[rel_inds[:, 0]]
    obj_scores1 = obj_scores.data[rel_inds[:, 1]]

    pred_scores_max, pred_classes_argmax = pred_scores.data[:, 1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1

    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    if rel_rank_scores is not None:
        rel_scores_argmaxed *= rel_rank_scores.data
    rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)

    rels = rel_inds[rel_scores_idx].cpu().numpy()
    pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()
    obj_scores_np = obj_scores.data.cpu().numpy()
    objs_np = obj_classes.data.cpu().numpy()
    boxes_out = boxes.data.cpu().numpy()

    if rel_rank_scores is not None:
        rel_rank_scores_np = rel_rank_scores[rel_scores_idx].data.cpu().numpy()
    else:
        rel_rank_scores_np = None

    if return_forest:
        assert forest is not None

    if im_inds is None:
        return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted, rel_rank_scores_np, gt_boxes, gt_classes, gt_rels, \
               forest if return_forest else None
    else:
        return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted, rel_rank_scores_np, gt_boxes, gt_classes, gt_rels, \
               forest if return_forest else None, im_inds.cpu().numpy()


def filter_dets_for_gcn_caption(im_inds, region_feats, obj_scores, obj_classes, rel_inds, pred_scores,
                                rel_rank_scores=None,
                                seq_labels=None, mask_labels=None, coco_ids=None):
    num_box = obj_classes.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 3
    assert pred_scores.size(0) == num_rel

    obj_scores0 = obj_scores.data[rel_inds[:, 1]]
    obj_scores1 = obj_scores.data[rel_inds[:, 2]]

    pred_scores_max, pred_classes_argmax = pred_scores.data[:, 1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1

    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    if rel_rank_scores is not None:
        rel_scores_argmaxed *= rel_rank_scores.data

    # split the relations according to image
    rel_im_inds = rel_inds[:, 0]

    rels = []
    pred_classes = []
    for i, s, e in enumerate_by_image(rel_im_inds):
        rels_i = rel_inds[s:e, :]
        pred_classes_argmax_i = pred_classes_argmax[s:e]
        rel_scores_argmaxed_i = rel_scores_argmaxed[s:e]
        rel_scores_vs_i, rel_scores_idx_i = torch.sort(rel_scores_argmaxed_i.view(-1), dim=0, descending=True)

        rels_i = rels_i[rel_scores_idx_i]
        pred_classes_argmax_i = pred_classes_argmax_i[rel_scores_idx_i]

        rels.append(rels_i)
        pred_classes.append(pred_classes_argmax_i)
    rels = torch.cat(rels, 0)
    pred_classes = torch.cat(pred_classes, 0)

    return im_inds, region_feats, pred_classes, rels, obj_classes.data, seq_labels, mask_labels, coco_ids


def filter_dets_for_caption(boxes, obj_scores, obj_classes, rel_inds, pred_scores, rel_feats, image_fmap,
                            rel_rank_scores=None,
                            seq_labels=None, mask_labels=None, coco_ids=None):
    """
        Filters detections....
        :param boxes: [num_box, 4]
        :param obj_scores: [num_box] probabilities for the scores
        :param obj_classes: [num_box] class labels for the topk
        :param rel_inds: [num_rel, 3] TENSOR consisting of (rel_im_inds, box_ind0, box_ind1)
        :param pred_scores: [num_rel, num_predicates]
        :param use_nms: True if use NMS to filter dets.
        :return:
        boxes: FloatTensor
        obj_classes: FloatTensor
        rels: LongTensor, [num_rel, 3]
        pred_classes: LongTensor, [num_rel,]
        rel_feats_all: FloatTensor, [num_rel, 4096]
        seq_labels: [num_img*5, 19], [im_inds, <start>, seq labels, <end>, 0, 0, ...]
        mask_labels: [num_img*5, 19], [im_inds, 1, 1, ..., {1 for <end>}, 0, 0, ...]

        """
    if boxes.dim() != 2:
        raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

    num_box = boxes.size(0)
    assert obj_scores.size(0) == num_box

    assert obj_classes.size() == obj_scores.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 3
    assert pred_scores.size(0) == num_rel

    obj_scores0 = obj_scores.data[rel_inds[:, 1]]
    obj_scores1 = obj_scores.data[rel_inds[:, 2]]

    pred_scores_max, pred_classes_argmax = pred_scores.data[:, 1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1

    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    if rel_rank_scores is not None:
        rel_scores_argmaxed *= rel_rank_scores.data

    # split the relations according to image
    rel_im_inds = rel_inds[:, 0]

    rels = []
    rel_feats_all = []
    pred_classes = []
    for i, s, e in enumerate_by_image(rel_im_inds):
        rels_i = rel_inds[s:e, :]
        pred_classes_argmax_i = pred_classes_argmax[s:e]
        rel_feats_i = rel_feats[s:e, :]
        rel_scores_argmaxed_i = rel_scores_argmaxed[s:e]
        rel_scores_vs_i, rel_scores_idx_i = torch.sort(rel_scores_argmaxed_i.view(-1), dim=0, descending=True)

        rels_i = rels_i[rel_scores_idx_i]
        pred_classes_argmax_i = pred_classes_argmax_i[rel_scores_idx_i]
        rel_feats_i = rel_feats_i[rel_scores_idx_i]

        rels.append(rels_i)
        rel_feats_all.append(rel_feats_i)
        pred_classes.append(pred_classes_argmax_i)
    rels = torch.cat(rels, 0)
    rel_feats_all = torch.cat(rel_feats_all, 0)
    pred_classes = torch.cat(pred_classes, 0)

    return boxes, obj_classes, rels, Variable(
        pred_classes), rel_feats_all, image_fmap, seq_labels, mask_labels, coco_ids
