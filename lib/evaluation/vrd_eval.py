# ---------------------------------------------------------------
# vrd_eval.py
# Set-up time: 2020/2/28 下午3:00
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------


"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
from functools import reduce
from lib.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
import itertools
from config import MODES

np.set_printoptions(precision=3)

from collections import defaultdict

"""
For Visual Relation Detection, Temporarily for SGdet(RelDet) and PhrDet
"""


class VRDEvaluator:
    def __init__(self, mode, multiple_preds=1, num_predicates=50):
        """
        :param mode:
        :param multiple_preds: the number of edges allowed between two nodes, 1 for graph constraints, typical is 10, 70
        :param num_predicates:
        """
        self.result_dict = {}

        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}

        self.multiple_preds = multiple_preds
        self.num_predicates = num_predicates

    @classmethod
    def vrd_modes(cls, **kwargs):
        # change the original preddet to sgdet
        evaluators = {'{}_{}'.format(m, multiple_preds): cls(mode=m, multiple_preds=multiple_preds, **kwargs) for
                      m, multiple_preds in
                      itertools.product(['sgdet', 'phrdet'], [1, 10, 70])}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_entry, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                                 iou_thresh=iou_thresh, multiple_preds=self.multiple_preds,
                                 num_predicates=self.num_predicates)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('======================' + self.mode + '_' + str(self.multiple_preds) + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
            if self.mode + '_recall_hit' in self.result_dict and self.mode + '_recall_count' in self.result_dict:
                avg = 0
                for idx in range(self.num_predicates):
                    tmp_avg = float(self.result_dict[self.mode + '_recall_hit'][k][idx + 1]) / float(
                        self.result_dict[self.mode + '_recall_count'][k][idx + 1] + 1e-10)
                    avg += tmp_avg
                    # print(str(idx+1), ' ', tmp_avg)
                print('cls avg', avg / (1.0 * self.num_predicates))
                print('total R', float(self.result_dict[self.mode + '_recall_hit'][k][0]) / float(
                    self.result_dict[self.mode + '_recall_count'][k][0] + 1e-10))


def evaluate_from_dict(gt_entry, pred_entry, mode, eval_result_dict,
                       multiple_preds=1, num_predicates=50, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict:
    :param viz_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']

    if mode == 'predcls':
        pred_boxes = gt_boxes
        pred_classes = gt_classes
        obj_scores = np.ones(gt_classes.shape[0])
    elif mode == 'sgcls':
        pred_boxes = gt_boxes
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'sgdet' or mode == 'phrdet':
        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in eval_result_dict[mode + '_recall']:
                eval_result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:, 1:])
        rel_scores_sorted[:, 1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:, 0]], rel_scores_sorted[:, 1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in eval_result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            eval_result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    else:
        raise ValueError('invalid mode')

    if multiple_preds > 1:
        if multiple_preds == rel_scores.shape[1] - 1: # all predicates
            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
            overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
            score_inds = argsort_desc(overall_scores)[:100]
            pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1] + 1))
            predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1] + 1]
        else:
            # between 1 and all predictes
            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1) # Nr
            overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]  # Nr, 70
            # sort predicate scores for each pair
            sorted_predicates_idx = np.argsort(-overall_scores, axis=1)[:, :multiple_preds]  # Nr, multiple_preds
            sorted_predicates_scores = np.sort(overall_scores, axis=1)[:, ::-1][:, :multiple_preds]
            score_inds = argsort_desc(sorted_predicates_scores)[:100]
            pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], sorted_predicates_idx[score_inds[:, 0], score_inds[:, 1]] + 1))
            predicate_scores = rel_scores[score_inds[:, 0], sorted_predicates_idx[score_inds[:, 0], score_inds[:, 1]] + 1]
    else:
        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        predicate_scores = rel_scores[:, 1:].max(1)

    RES_pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
        gt_rels, gt_boxes, gt_classes,
        pred_rels, pred_boxes, pred_classes,
        predicate_scores, obj_scores, phrdet=mode == 'phrdet',
        **kwargs)

    pred_to_gt = RES_pred_to_gt
    result_dict = eval_result_dict

    for k in result_dict[mode + '_recall']:

        match = reduce(np.union1d, pred_to_gt[:k])

        for idx in range(len(match)):
            local_label = gt_rels[int(match[idx]), 2]
            if (mode + '_recall_hit') not in result_dict:
                result_dict[mode + '_recall_hit'] = {}
            if k not in result_dict[mode + '_recall_hit']:
                result_dict[mode + '_recall_hit'][k] = [0] * (num_predicates + 1)
            result_dict[mode + '_recall_hit'][k][int(local_label)] += 1
            result_dict[mode + '_recall_hit'][k][0] += 1

        for idx in range(gt_rels.shape[0]):
            local_label = gt_rels[idx, 2]
            if (mode + '_recall_count') not in result_dict:
                result_dict[mode + '_recall_count'] = {}
            if k not in result_dict[mode + '_recall_count']:
                result_dict[mode + '_recall_count'][k] = [0] * (num_predicates + 1)
            result_dict[mode + '_recall_count'][k][int(local_label)] += 1
            result_dict[mode + '_recall_count'][k][0] += 1

        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)

    return RES_pred_to_gt, pred_5ples, rel_scores


###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:, :2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,1])
    assert np.all(pred_rels[:, 2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:, 2], pred_rels[:, :2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    scores_overall = relation_scores.prod(1)
    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )
    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:, :2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                          gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def _compute_subobj_matches(gt_triplets, pred_triplets,
                            gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(np.column_stack((gt_triplets[:, 0:1], gt_triplets[:, 2:3])),
                         np.column_stack((pred_triplets[:, 0:1], pred_triplets[:, 2:3])))
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
