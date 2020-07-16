import torch
import numpy as np
from lib.pytorch_misc import enumerate_by_image, gather_nd, random_choose
from lib.pytorch_misc import diagonal_inds, to_variable
from lib.fpn.co_nms.functions.co_nms import apply_co_nms


@to_variable
def rel_proposal(rois, scores, nms_thresh=0.7, pre_nms_topn=60000, post_nms_topn=10000):
    """

    :param rois: N, 5 [im_inds, x1, y1, x2, y2]
    :param scores: N, N
    :param nms_thresh:
    :param pre_nms_topn:
    :param post_nms_topn:
    :return: all_pair_inds [im_inds, ind1, ind2]
    """

    im_inds = rois[:, 0].long()  # N
    num_im = im_inds[-1] + 1
    is_cand = (im_inds[:, None] == im_inds[None])  # N, N
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0
    rel_inds = torch.nonzero(is_cand)  # Npair, 2

    # get the scores
    scores = scores[rel_inds[:, 0], rel_inds[:, 1]]  # Npair

    # compute the inds for each pair ind, that is how many rel proposals for each image
    num_pairs = np.zeros(num_im + 1).astype(np.int32)
    id_to_iminds = {}
    for i, s, e in enumerate_by_image(im_inds):
        num_pairs[i + 1] = (e - s) * (e - s - 1)
        id_to_iminds[i] = im_inds[s]
    cumsum_num_pairs = np.cumsum(num_pairs).astype(np.int32)

    all_pair_inds = []
    all_pair_scores = []
    for i in range(1, num_im+1):
        start = cumsum_num_pairs[i - 1]
        scores_i = scores[start:(start+num_pairs[i])]
        pair_inds_i = rel_inds[start:(start+num_pairs[i]), :]

        _, order = torch.sort(scores_i, descending=True)

        if pre_nms_topn > 0 and pre_nms_topn < num_pairs[i]:
            order_single = order[:pre_nms_topn]
        else:
            order_single = order

        pair_inds_single = pair_inds_i[order_single, :]
        scores_single = scores_i[order_single].view(-1, 1)

        proposal_subject = rois[pair_inds_single[:, 0], :][:, 1:]
        proposal_object = rois[pair_inds_single[:, 1], :][:, 1:]

        # the box pairs has been sorted
        keep, num_out = apply_co_nms(torch.cat((proposal_subject, proposal_object), 1), nms_thresh, post_nms_topn)

        pair_inds_single = pair_inds_single[keep, :]
        scores_single = scores_single[keep, :].view(-1)
        im_inds_i = torch.LongTensor([id_to_iminds[i-1]] * num_out).view(-1, 1).cuda(pair_inds_single.get_device())
        pair_inds_single = torch.cat((im_inds_i, pair_inds_single), 1)

        all_pair_inds.append(pair_inds_single)
        all_pair_scores.append(scores_single)

    all_pair_inds = torch.cat(all_pair_inds, 0)
    # sort by rel
    _, perm = torch.sort(all_pair_inds[:, 0] * (rois.size(0) ** 2) + all_pair_inds[:, 1] * rois.size(0) + all_pair_inds[:, 2])
    all_pair_inds = all_pair_inds[perm].contiguous()
    all_pair_scores = torch.cat(all_pair_scores, 0)[perm].contiguous()
    return all_pair_inds, all_pair_scores



