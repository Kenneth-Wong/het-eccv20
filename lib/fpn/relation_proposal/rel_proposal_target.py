import torch
import numpy as np
from lib.pytorch_misc import enumerate_by_image, gather_nd, random_choose, intersect_2d
from lib.pytorch_misc import diagonal_inds, to_variable
from lib.fpn.co_nms.functions.co_nms import apply_co_nms
from config import RELS_BATCHSIZE, REL_FG_FRACTION
from lib.fpn.box_utils import co_bbox_overlaps
from collections import defaultdict
import random

@to_variable
def rel_proposal_target(rois, rel_proposal_inds, gt_boxes, gt_classes, gt_rels, image_offset, mode):
    """
    Assign the tareget for each proposal pairs.
    When the mode is predcls or sgcls, the target is directly obtained by comparing with gt_rel.
    When the mode is sgdet, the target is sampled by firstly compute iou with gt_pairs
    :param rois:
    :param rel_proposal_inds: [im_ind, ind1, ind2]
    :param gt_boxes:
    :param image_offset:
    :param mode:
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

    rels_to_gt = []
    num_gt_rels_seen = 0

    if mode in ('predcls', 'sgcls'):
        rel_proposal_inds_np = rel_proposal_inds.cpu().numpy()
        fg_rels_np = fg_rels.cpu().numpy()  ## Ngtp, 4

        # locate the proposal
        locate_inds = np.where(intersect_2d(rel_proposal_inds_np, fg_rels_np[:, :-1]))
        proposal_to_gt = defaultdict(list)
        for ind in zip(*locate_inds):
            proposal_to_gt[ind[0]].append(ind[1])
        for k, v in proposal_to_gt.items():
            v0 = v[0] if len(v) == 1 else np.random.choice(v)
            proposal_to_gt[k] = v0



        fg_proposal_inds = np.array(list(proposal_to_gt.keys())).astype(np.int32)
        bg_proposal_inds = np.array(list(set(list(range(rel_proposal_inds_np.shape[0]))) - set(list(proposal_to_gt.keys())))).astype(np.int32)

        rels_to_gt = np.ones(fg_proposal_inds.shape[0] + bg_proposal_inds.shape[0], dtype=np.int64) * -1
        if len(fg_proposal_inds) > 0:
            rels_to_gt[fg_proposal_inds] = np.array([proposal_to_gt[ind] for ind in fg_proposal_inds])

        num_fg = min(fg_proposal_inds.size, int(RELS_BATCHSIZE * REL_FG_FRACTION * num_im))
        if num_fg < fg_proposal_inds.size:
            fg_proposal_inds = np.random.choice(fg_proposal_inds, num_fg, replace=False)
        num_bg = min(bg_proposal_inds.size if bg_proposal_inds.size else 0, int(RELS_BATCHSIZE * num_im) - num_fg)
        if num_bg < bg_proposal_inds.size:
            bg_proposal_inds = np.random.choice(bg_proposal_inds, num_bg, replace=False)

        if len(fg_proposal_inds) == 0:
            bg_labels = np.zeros(bg_proposal_inds.size)
            bg_rel_labels = np.hstack((rel_proposal_inds_np[bg_proposal_inds], bg_labels[:, None]))
            proposal_labels = bg_rel_labels
        else:
            fg_labels = np.array([fg_rels[proposal_to_gt[ind], -1] for ind in fg_proposal_inds])
            fg_rel_labels = np.hstack((rel_proposal_inds_np[fg_proposal_inds], fg_labels[:, None]))

            bg_labels = np.zeros(bg_proposal_inds.size)
            bg_rel_labels = np.hstack((rel_proposal_inds_np[bg_proposal_inds], bg_labels[:, None]))
            proposal_labels = np.vstack((fg_rel_labels, bg_rel_labels))

            rels_to_gt = np.hstack((rels_to_gt[fg_proposal_inds], rels_to_gt[bg_proposal_inds]))

        proposal_labels = torch.LongTensor(proposal_labels).cuda(gt_rels.get_device())
        rels_to_gt = torch.LongTensor(rels_to_gt).cuda(gt_rels.get_device())
    else:
        assert mode == 'sgdet'

        gt_box_pairs = torch.cat((gt_boxes[fg_rels[:, 1]], gt_boxes[fg_rels[:, 2]]), 1)
        rel_proposal_pairs = torch.cat((rois[:, 1:][rel_proposal_inds[:, 0]], rois[:, 1:][rel_proposal_inds[:, 1]]), 1)

        num_pairs = np.zeros(num_im + 1).astype(np.int32)
        for i, s, e in enumerate_by_image(rel_proposal_inds[:, 0]):
            num_pairs[i + 1] = e - s

        cumsum_num_pairs = np.cumsum(num_pairs).astype(np.int32)
        fg_rel_per_image = int(RELS_BATCHSIZE * REL_FG_FRACTION)

        proposal_labels = []
        gt_rel_labels = fg_rels[:, -1].contiguous().view(-1)
        for i in range(1, num_im + 1):
            rel_proposal_inds_i = rel_proposal_inds[cumsum_num_pairs[i - 1]:cumsum_num_pairs[i]]
            rel_proposal_pairs_i = rel_proposal_pairs[cumsum_num_pairs[i - 1]:cumsum_num_pairs[i]]
            gt_box_pairs_i = gt_box_pairs[torch.nonzero(fg_rels[:, 0] == (i - 1)).view(-1)]

            gt_box_pairs_label_i = gt_rel_labels[torch.nonzero(fg_rels[:, 0] == (i - 1)).view(-1)].view(-1).contiguous()

            overlaps = co_bbox_overlaps(rel_proposal_pairs_i, gt_box_pairs_i)  # Np, Ngtp
            max_overlaps, gt_assignment = torch.max(overlaps, 1)  # Np
            fg_inds = torch.nonzero(max_overlaps >= 0.5).view(-1)
            fg_num = fg_inds.numel()

            bg_inds = torch.nonzero((max_overlaps < 0.5) & (max_overlaps >= 0.0)).view(-1)
            bg_num = bg_inds.numel()

            rels_to_gt_i = torch.LongTensor(rel_proposal_pairs_i.shape[0]).fill(-1).cuda(gt_rels.get_device())
            rels_to_gt_i[fg_inds] = gt_assignment[fg_inds] + num_gt_rels_seen

            if fg_num > 0 and bg_num > 0:
                fg_this_image = min(fg_rel_per_image, fg_num)
                rand_num = torch.from_numpy(np.random.permutation(fg_num)).long().cuda()
                fg_inds = fg_inds[rand_num[:fg_this_image]]

                # sampling bg
                bg_this_image = RELS_BATCHSIZE - fg_this_image
                rand_num = np.floor(np.random.rand(bg_this_image) * bg_num)
                rand_num = torch.from_numpy(rand_num).long().cuda()
                bg_inds = bg_inds[rand_num]

                rels_to_gt_i = torch.cat((rels_to_gt_i[fg_inds], rels_to_gt_i[bg_inds]), 0)

            elif fg_num > 0 and bg_num == 0:
                rand_num = np.floor(np.random.rand(RELS_BATCHSIZE) * fg_num)
                rand_num = torch.from_numpy(rand_num).long().cuda()
                fg_inds = fg_inds[rand_num]
                fg_this_image = RELS_BATCHSIZE
                bg_this_image = 0
                rels_to_gt_i = rels_to_gt_i[fg_inds]
            elif bg_num > 0 and fg_num == 0:
                # sampling bg
                # rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(RELS_BATCHSIZE) * bg_num)
                rand_num = torch.from_numpy(rand_num).long().cuda()

                bg_inds = bg_inds[rand_num]
                bg_this_image = RELS_BATCHSIZE
                fg_this_image = 0
                rels_to_gt_i = rels_to_gt_i[bg_inds]
            else:
                import pdb
                pdb.set_trace()

            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            rel_proposal_inds_i = rel_proposal_inds_i[keep_inds]
            labels_i = gt_box_pairs_label_i[gt_assignment[keep_inds]]
            if fg_this_image < labels_i.size(0):
                labels_i[fg_this_image:] = 0
            rels_to_gt.append(rels_to_gt_i)
            num_gt_rels_seen += gt_box_pairs_i.shape[0]
            #try:
            #    labels_i[fg_this_image:] = 0
            #except ValueError:
            #    print(labels_i)
            #    print(fg_this_image)
            #    import pdb
            #    pdb.set_trace()
            proposal_labels.append(torch.cat((rel_proposal_inds_i, labels_i[:, None]), 1))
        proposal_labels = torch.cat(proposal_labels, 0)
        rels_to_gt = torch.cat(rels_to_gt, 0)

    # sort
    _, perm = torch.sort(
        proposal_labels[:, 0] * (rois.size(0) ** 2) + proposal_labels[:, 1] * rois.size(0) + proposal_labels[:, 2])
    proposal_labels = proposal_labels[perm].contiguous()
    rels_to_gt = rels_to_gt[perm].contiguous()

    return proposal_labels, rels_to_gt








