import torch
import numpy as np
import torch.nn as nn
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.fpn.box_utils import bbox_overlaps, center_size
from torch.nn.utils.rnn import PackedSequence
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image
from torch.autograd import Variable


def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1] + 1
    rois_per_image = scores.new(num_im)
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


class RankingContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self,
                 hidden_dim=512, rel_visual_dim=4096, rel_pos_inp_dim=6, rel_pos_dim=256,
                 dropout_rate=0.2, nl_ranking_layer=4, order='leftright', sal_input='both'):
        super(RankingContext, self).__init__()


        self.hidden_dim = hidden_dim
        self.rel_pair_dim = rel_visual_dim
        self.rel_pos_inp_dim = rel_pos_inp_dim
        self.rel_pos_dim = rel_pos_dim

        self.dropout_rate = dropout_rate


        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order
        self.nl_ranking_layer = nl_ranking_layer

        self.pos_proj = nn.Linear(self.rel_pos_inp_dim, self.rel_pos_dim)

        self.ranking_ctx_rnn = AlternatingHighwayLSTM(
                input_size=self.rel_pair_dim + self.rel_pos_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_ranking_layer,
                recurrent_dropout_probability=dropout_rate)

        assert sal_input in ('both', 'sal', 'area', 'empty')
        self.sal_input = sal_input

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:,2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:,0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)


    def forward(self, rel_union_rep, pos_rep, pair_im_inds, union_boxes):
        """
        Forward pass through the object and edge context
        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """
        rel_input = torch.cat((rel_union_rep, self.pos_proj(pos_rep)), 1)
        perm, inv_perm, ls_transposed = self.sort_rois(pair_im_inds, None, union_boxes[:, 1:])

        rel_inpunt_rep = rel_input[perm].contiguous()
        rel_input_packed = PackedSequence(rel_inpunt_rep, ls_transposed)

        rel_rank_rep = self.ranking_ctx_rnn(rel_input_packed)[0][0]
        rel_rank_rep = rel_rank_rep[inv_perm].contiguous()

        return rel_rank_rep
