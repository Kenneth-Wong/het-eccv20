import torch
import numpy as np
from lib.pytorch_misc import enumerate_by_image, gather_nd, random_choose
from lib.pytorch_misc import diagonal_inds, to_variable
from lib.fpn.co_nms.functions.co_nms import apply_co_nms
from config import RELPN_BATCHSIZE, RELPN_FG_FRACTION
from lib.fpn.box_utils import co_bbox_overlaps

@to_variable
def rel_anchor_target(rois, gt_boxes, gt_classes, scores, gt_rels, image_offset):
    """
    use all roi pairs and sample some pairs to train relation proposal module
    Note: ONLY for mode SGDET!!!!
    rois are from RPN,
    We take the CO_Overlap strategy from Graph-RCNN to sample fg and bg rels
    :param rois: N, 5
    :param scores: N, N
    :param gt_rels:
    :return:
    """
    im_inds = rois[:, 0].long()
    num_im = im_inds[-1] + 1

    # Offset the image indices in fg_rels to refer to absolute indices (not just within img i)
    fg_rels = gt_rels.clone()
    fg_rels[:, 0] -= image_offset
    offset = {}
    for i, s, e in enumerate_by_image(gt_classes[:, 0]):
        offset[i] = s
    for i, s, e in enumerate_by_image(fg_rels[:, 0]):
        fg_rels[s:e, 1:3] += offset[i]

    gt_box_pairs = torch.cat((gt_boxes[fg_rels[:, 1]], gt_boxes[fg_rels[:, 2]]), 1)  # Ngtp, 8

    # get all potential pairs
    is_cand = (im_inds[:, None] == im_inds[None])
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0

    all_pair_inds = torch.nonzero(is_cand)
    all_box_pairs = torch.cat((rois[:, 1:][all_pair_inds[:, 0]], rois[:, 1:][all_pair_inds[:, 1]]), 1)

    num_pairs = np.zeros(num_im + 1).astype(np.int32)
    id_to_iminds = {}
    for i, s, e in enumerate_by_image(im_inds):
        num_pairs[i + 1] = (e - s) * (e - s - 1)
        id_to_iminds[i] = im_inds[s]
    cumsum_num_pairs = np.cumsum(num_pairs).astype(np.int32)

    all_rel_inds = []
    for i in range(1, num_im+1):
        all_pair_inds_i = all_pair_inds[cumsum_num_pairs[i-1]:cumsum_num_pairs[i]]
        all_box_pairs_i = all_box_pairs[cumsum_num_pairs[i-1]:cumsum_num_pairs[i]]
        gt_box_pairs_i = gt_box_pairs[torch.nonzero(fg_rels[:, 0] == (i - 1)).view(-1)]
        labels = gt_rels.new(all_box_pairs_i.size(0)).fill_(-1)

        overlaps = co_bbox_overlaps(all_box_pairs_i, gt_box_pairs_i)  ## Np, Ngtp
        max_overlaps, argmax_overlaps = torch.max(overlaps, 1)  ## Np
        gt_max_overlaps, _ = torch.max(overlaps, 0)  ## Ngtp

        labels[max_overlaps < 0.15] = 0
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5

        # fg rel: for each gt pair, the max overlap anchor is fg
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(1, -1).expand_as(overlaps)), 1)  # Np
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        # fg rel: above thresh
        labels[max_overlaps >= 0.25] = 1

        num_fg = int(RELPN_BATCHSIZE * RELPN_FG_FRACTION)
        sum_fg = torch.sum((labels == 1).int())
        sum_bg = torch.sum((labels == 0).int())

        if sum_fg > num_fg:
            fg_inds = torch.nonzero(labels == 1).view(-1)
            rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
            disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
            labels[disable_inds] = -1
        num_bg = RELPN_BATCHSIZE - torch.sum((labels == 1).int())

        if sum_bg > num_bg:
            bg_inds = torch.nonzero(labels == 0).view(-1)
            rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
            disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
            labels[disable_inds] = -1

        keep_inds = torch.nonzero(labels >= 0).view(-1)
        labels = labels[keep_inds]
        all_pair_inds_i = all_pair_inds_i[keep_inds]

        im_inds_i = torch.LongTensor([id_to_iminds[i - 1]] * keep_inds.size(0)).view(-1, 1).cuda(all_pair_inds.get_device())
        all_pair_inds_i = torch.cat((im_inds_i, all_pair_inds_i, labels.view(-1, 1)), 1)
        all_rel_inds.append(all_pair_inds_i)

    all_rel_inds = torch.cat(all_rel_inds, 0)
    # sort by rel
    _, perm = torch.sort(
        all_rel_inds[:, 0] * (rois.size(0) ** 2) + all_rel_inds[:, 1] * rois.size(0) + all_rel_inds[:, 2])
    all_rel_inds = all_rel_inds[perm].contiguous()
    return all_rel_inds














