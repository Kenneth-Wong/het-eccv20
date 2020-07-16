"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms
from lib.fpn.co_nms.functions.co_nms import apply_co_nms
from lib.fpn.relation_proposal.rel_proposal import rel_proposal
from lib.fpn.relation_proposal.rel_anchor_target import rel_anchor_target
from lib.fpn.relation_proposal.rel_proposal_target import rel_proposal_target
# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from lib.fpn.box_utils import bbox_overlaps, center_size, bbox_overlaps_pair
from lib.get_union_boxes import UnionBoxesAndFeats, union_regions
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, \
    Flattener, intersect_2d, keyrel_loss, FocalLoss, get_area_maps
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets, filter_dets_for_caption, filter_dets_for_gcn_caption
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math
from config import IM_SCALE, ROOT_PATH, CO_OCCOUR_PATH, CO_OCCOUR_PATH_VG200

from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.lstm.RankingContext import RankingContext
# import tree lstm
from lib.tree_lstm import hrtree_lstm, gen_tree, gen_hrtree, tree_utils
from lib.tree_lstm.draw_tree import draw_tree_region, draw_tree_region_v2
from lib.tree_lstm.decoder_tree_lstm import DecoderTreeLSTM
from lib.tree_lstm.decoder_hrtree_lstm import DecoderHrTreeLSTM
from lib.tree_lstm.graph_to_tree import graph_to_trees, arbitraryForest_to_biForest

from lib.lstm.attention_rnn import AttentionRNN
from lib.lstm.plain_rnn import PlainRNN
import itertools
from copy import deepcopy


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


MODES = ('sgdet', 'sgcls', 'predcls')


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, hidden_dim=256, obj_dim=2048,
                 nl_obj=2, nl_edge=2, dropout_rate=0.2, order='confidence',
                 pick_parent='area',
                 isc_thresh=0.9,
                 pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True,
                 draw_tree=False,
                 saliency=False,
                 use_depth=False):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.nl_obj = nl_obj
        self.nl_edge = nl_edge

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge
        self.draw_tree = draw_tree

        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order

        assert pick_parent in ('area', 'isc', 'depth')
        self.pick_parent = pick_parent

        self.isc_thresh = isc_thresh

        self.saliency = saliency

        self.use_depth = use_depth

        # EMBEDDINGS
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()
        self.virtual_node_embed = nn.Embedding(1, self.embed_dim)  # used to encode Root Node

        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
        ])

        # whether draw tree
        if self.draw_tree:
            self.draw_tree_count = 0
            self.draw_tree_max = 600

        if self.nl_obj > 0:
            self.obj_tree_lstm = hrtree_lstm.MultiLayer_BTreeLSTM(self.obj_dim + self.embed_dim + 128, self.hidden_dim,
                                                                  self.nl_obj, dropout_rate,
                                                                  saliency)

            decoder_inputs_dim = self.hidden_dim
            if self.pass_in_obj_feats_to_decoder:
                decoder_inputs_dim += self.obj_dim + self.embed_dim

            self.decoder_tree_lstm = DecoderHrTreeLSTM(classes, embed_dim=100,  # embed_dim = self.embed_dim,
                                                       inputs_dim=decoder_inputs_dim,
                                                       hidden_dim=self.hidden_dim,
                                                       direction='backward',
                                                       dropout=dropout_rate,
                                                       pass_root=False)
        else:
            self.decoder_lin = nn.Linear(self.obj_dim + self.embed_dim + 128, self.num_classes)

        if self.nl_edge > 0:
            input_dim = self.embed_dim
            if self.nl_obj > 0:
                input_dim += self.hidden_dim
            if self.pass_in_obj_feats_to_edge:
                input_dim += self.obj_dim

            self.edge_tree_lstm = hrtree_lstm.MultiLayer_BTreeLSTM(input_dim, self.hidden_dim, self.nl_edge,
                                                                   dropout_rate)

        self.pooling_size = 7

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:, 2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:, 0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def sal_feature_map(self, sal_maps, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            sal_maps, rois).view(rois.shape[0], -1)
        return feature_pool  # num_rois, 49

    def edge_ctx(self, obj_feats, obj_preds, box_priors=None, forest=None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """

        # Only use hard embeddings
        obj_embed2 = self.obj_embed2(obj_preds)
        inp_feats = torch.cat((obj_embed2, obj_feats), 1)
        # use bidirectional tree lstm to update
        edge_ctx = self.edge_tree_lstm(forest, inp_feats, box_priors.shape[0])
        return edge_ctx

    def obj_ctx(self, obj_feats, obj_labels=None, box_priors=None, boxes_per_cls=None, forest=None, batch_size=0):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # use bidirectional tree lstm to update
        encoder_rep = self.obj_tree_lstm(forest, obj_feats, box_priors.shape[0])

        # Decode in order
        if self.mode != 'predcls':
            decode_feature = torch.cat((obj_feats, encoder_rep),
                                       1) if self.pass_in_obj_feats_to_decoder else encoder_rep
            obj_dists, obj_preds = self.decoder_tree_lstm(forest, decode_feature,
                                                          box_priors.shape[0],
                                                          labels=obj_labels if obj_labels is not None else None,
                                                          boxes_for_nms=boxes_per_cls if boxes_per_cls is not None else None,
                                                          batch_size=batch_size)
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = Variable(to_onehot(obj_preds.data[:-batch_size], self.num_classes))

        return obj_dists, obj_preds, encoder_rep

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None,
                gt_forest=None, image_rois=None, image_fmap=None, rel_labels=None, sal_maps=None, origin_img=None,
                depth_maps=None):
        """
        Forward pass through the object and edge context
        :param obj_priors: [obj_num, (x1,y1,x2,y2)], float cuda
        :param obj_fmaps:
        :param im_inds: [obj_num] long variable
        :param obj_labels:
        :param boxes:
        :return:
        """
        if self.mode == 'predcls':
            obj_logits = Variable(to_onehot(obj_labels.data, self.num_classes))

        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        batch_size = image_rois.shape[0]
        # pseudo box and image index: to encode virtual node into original inputs
        pseudo_box_priors = torch.cat((box_priors, image_rois[:, 1:].contiguous().data), 0)  # [obj_num + batch_size, 4]
        pseudo_im_inds = torch.cat((im_inds, image_rois[:, 0].contiguous().long().view(-1)),
                                   0)  # [obj_num + batch_size]
        pseudo_obj_fmaps = torch.cat((obj_fmaps.clone().detach(), image_fmap.detach()),
                                     0)  # [obj_num + batch_size, 4096]
        virtual_embed = self.virtual_node_embed.weight[0].view(1, -1).expand(batch_size, -1)
        pseudo_obj_embed = torch.cat((obj_embed, virtual_embed), 0)  # [obj_num + batch_size, embed_dim]
        if self.training or (self.mode == 'predcls'):
            pseudo_obj_labels = torch.cat(
                (obj_labels, Variable(torch.randn(1).fill_(0).cuda()).expand(batch_size).long().view(-1)), 0)
        else:
            pseudo_obj_labels = None

        if self.mode == 'sgdet':
            obj_distributions = F.softmax(obj_logits, dim=1)[:, 1:]
        else:
            obj_distributions = F.softmax(obj_logits[:, 1:], dim=1)
        pseudo_obj_distributions = torch.cat(
            (obj_distributions, Variable(torch.randn(batch_size, obj_distributions.shape[1]).fill_(0).cuda())), 0)

        # get sal maps
        if sal_maps is not None:
            pseudo_box_rois = Variable(torch.cat((pseudo_im_inds[:, None].float().data, pseudo_box_priors), 1))
            pseudo_sal_maps = self.sal_feature_map(sal_maps, pseudo_box_rois)  # N, 49
        else:
            pseudo_sal_maps = None

        # print('node_scores', node_scores.data.cpu().numpy())
        forest = gen_hrtree.generate_forest(pseudo_im_inds, Variable(pseudo_box_priors),
                                            torch.max(pseudo_obj_distributions, dim=1)[0],
                                            obj_labels, self.pick_parent, self.isc_thresh, child_order=self.order,
                                            sal_maps=pseudo_sal_maps, depth_maps=depth_maps)

        # arbitrary_forest, gen_tree_loss, entropy_loss = gen_tree.generate_forest(pseudo_im_inds, gt_forest, pair_scores, Variable(pseudo_box_priors), pseudo_obj_labels, self.use_rl_tree, self.training, self.mode)
        # forest = arbitraryForest_to_biForest(arbitrary_forest)

        pseudo_pos_embed = self.pos_embed(Variable(center_size(pseudo_box_priors)))
        obj_pre_rep = torch.cat((pseudo_obj_fmaps, pseudo_obj_embed, pseudo_pos_embed), 1)
        if self.nl_obj > 0:
            obj_dists2, obj_preds, obj_ctx = self.obj_ctx(
                obj_pre_rep,
                pseudo_obj_labels,
                pseudo_box_priors,
                boxes_per_cls,
                forest,
                batch_size
            )
        else:
            print('Error, No obj ctx')

        edge_ctx = None
        if self.nl_edge > 0:
            edge_ctx = self.edge_ctx(
                torch.cat((pseudo_obj_fmaps, obj_ctx), 1) if self.pass_in_obj_feats_to_edge else obj_ctx,
                obj_preds=obj_preds,
                box_priors=pseudo_box_priors,
                forest=forest,
            )

        # draw tree
        if self.draw_tree and (self.draw_tree_count < self.draw_tree_max):
            for tree_idx in range(len(forest)):
                draw_tree_region(forest[tree_idx], origin_img, self.draw_tree_count)
                draw_tree_region_v2(forest[tree_idx], origin_img, self.draw_tree_count, obj_preds)
                self.draw_tree_count += 1

        # remove virtual nodes
        return obj_dists2, obj_preds[:-batch_size], edge_ctx[:-batch_size], forest


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=1, nl_edge=2, nl_rel=4, use_resnet=False, order='confidence', pick_parent='area', isc_thresh=0.9,
                 thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.1, use_bias=True, use_tanh=True, use_encoded_box=True,
                 draw_tree=False,
                 limit_vision=True,
                 saliency=False,
                 need_relpn=False,
                 need_relrank=False,
                 relpn_embed_dim=0,
                 relpn_with_bbox_info=False,
                 use_CE=False,
                 dbname='vg',
                 sal_input='both',
                 use_depth=False,
                 has_grad=False,
                 use_dist=False,
                 return_forest=False,
                 need_caption=False,
                 gcn_caption=False):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.dbname = dbname

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.use_encoded_box = use_encoded_box
        self.draw_tree = draw_tree
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        # whether need relpn and relrank
        self.need_relpn = need_relpn
        self.need_relrank = need_relrank
        self.sal_input = sal_input

        self.use_depth = use_depth

        self.relpn_embed_dim = relpn_embed_dim
        self.relpn_with_bbox_info = relpn_with_bbox_info

        # use (Focal)CE or max-margin loss
        self.use_CE = use_CE

        # whether need saliency info
        self.need_saliency = saliency

        self.has_grad = has_grad
        self.use_dist = use_dist

        self.return_forest = return_forest

        self.need_caption = need_caption
        self.gcn_caption = gcn_caption

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )

        if self.need_relpn:
            self.relpn_head = RelPNHead(dim=512, num_classes=len(self.classes), embed_dim=self.relpn_embed_dim,
                                        pos_dim=(4 if self.relpn_with_bbox_info else 0))
            if self.relpn_embed_dim > 0:
                embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.relpn_embed_dim)
                self.relpn_embed = nn.Embedding(self.num_classes, self.relpn_embed_dim)
                self.relpn_embed.weight.data = embed_vecs.clone()

        # another set of parameters for relrank
        if self.need_relrank:
            self.rank_context = RankingContext(hidden_dim=512, rel_visual_dim=4096, rel_pos_inp_dim=6,
                                                    rel_pos_dim=256, dropout_rate=rec_dropout, nl_ranking_layer=nl_rel,
                                                    order=order, sal_input=sal_input)
            self.rank_visual_proj = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)])

        #"""
        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         obj_dim=self.obj_dim,
                                         nl_obj=nl_obj, nl_edge=nl_edge, dropout_rate=rec_dropout,
                                         order=order,
                                         pick_parent=pick_parent,
                                         isc_thresh=isc_thresh,
                                         pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                                         pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge,
                                         draw_tree=self.draw_tree,
                                         saliency=False,
                                         use_depth=self.use_depth)
        #"""

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096,
                         pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)  # for union box
            self.roi_fmap_obj = load_vgg(use_dropout=False, pretrained=False).classifier  # for object

        ###################################
        self.post_lstm = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
        self.post_lstm.bias.data.zero_()
        self.post_cat.weight = torch.nn.init.xavier_normal(self.post_cat.weight, gain=1.0)
        self.post_cat.bias.data.zero_()

        if self.use_encoded_box:
            # encode spatial info
            self.encode_spatial_1 = nn.Linear(32, 512)
            self.encode_spatial_2 = nn.Linear(512, self.pooling_dim)

            self.encode_spatial_1.weight.data.normal_(0, 1.0)
            self.encode_spatial_1.bias.data.zero_()
            self.encode_spatial_2.weight.data.normal_(0, 0.1)
            self.encode_spatial_2.bias.data.zero_()

        if nl_edge == 0:
            self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim * 2)
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias(dbname=dbname)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def rank_visual_rep(self, features, rois, pair_inds):
        assert pair_inds.size(1) == 2
        uboxes = self.rankLSTM_union_boxes(features, rois, pair_inds)
        return self.rankLSTM_roi_fmap(uboxes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def extract_rel_inds_from_tree(self, tree, rel_inds):
        num_child = len(tree.children)
        if num_child == 0:
            return
        children = tree.children
        all_indexes = []
        if not tree.is_root:
            all_indexes.append(tree.index)
        for c in children:
            all_indexes.append(c.index)
        for item in itertools.product(all_indexes, all_indexes):
            if item[0] == item[1]:
                continue
            rel_inds.append([item[0], item[1]])

        for i in range(num_child):
            self.extract_rel_inds_from_tree(children[i], rel_inds)

    def get_hr_rel_inds(self, rel_labels, im_inds, box_priors, forest=None):
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            assert forest is not None
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data, box_priors.data) > 0)

            rel_inds = []
            for tree in forest:
                self.extract_rel_inds_from_tree(tree, rel_inds)
            rel_map = rel_cands.new(im_inds.data.shape[0], im_inds.data.shape[0]).fill_(0)
            rel_inds = torch.LongTensor(rel_inds).cuda()
            rel_map.view(-1)[rel_inds[:, 0] * im_inds.data.shape[0] + rel_inds[:, 1]] = 1
            # rel_map[rel_inds[:, 0], rel_inds[:, 1]] = 1
            rel_cands = rel_cands & rel_map

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def obj_sal_value(self, sal_maps, rois):
        salmap_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(sal_maps, rois)
        num_rois = salmap_pool.size(0)
        sal_value = F.avg_pool2d(salmap_pool, kernel_size=self.pooling_size, stride=1).view(num_rois, -1)
        return sal_value  # N, 1

    def get_rel_label(self, im_inds, gt_rels, rel_inds):
        np_im_inds = im_inds.data.cpu().numpy()
        np_gt_rels = gt_rels.long().data.cpu().numpy()
        np_rel_inds = rel_inds.long().cpu().numpy()

        num_obj = int(im_inds.shape[0])
        sub_id = np_rel_inds[:, 1]
        obj_id = np_rel_inds[:, 2]
        select_id = sub_id * num_obj + obj_id

        count = 0
        offset = 0
        slicedInds = np.where(np_im_inds == count)[0]

        label = np.array([0] * num_obj * num_obj, dtype=int)
        while (len(slicedInds) > 0):
            slice_len = len(slicedInds)
            selectInds = np.where(np_gt_rels[:, 0] == count)[0]
            slicedRels = np_gt_rels[selectInds, :]
            flattenID = (slicedRels[:, 1] + offset) * num_obj + (slicedRels[:, 2] + offset)
            slicedLabel = slicedRels[:, 3]

            label[flattenID] = slicedLabel

            count += 1
            offset += slice_len
            slicedInds = np.where(np_im_inds == count)[0]

        return Variable(torch.from_numpy(label[select_id]).long().cuda())

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                sal_maps=None, key_rels=None, depth_maps=None, seq_labels=None, mask_labels=None, coco_ids=None, return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is not None:
            assert self.mode in ('predcls', 'sgcls')

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels, fg_rel_labels, result.rels_to_gt = rel_assignments(im_inds.data, boxes.data,
                                                                                  result.rm_obj_labels.data,
                                                                                  gt_boxes.data, gt_classes.data,
                                                                                  gt_rels.data,
                                                                                  image_offset, filter_non_overlap=True,
                                                                                  num_sample_per_gt=1)

            # if self.training and (not self.use_rl_tree):
            # generate arbitrary forest according to graph
        #    arbitrary_forest = graph_to_trees(self.co_occour, result.rel_labels, gt_classes)
        # else:
        arbitrary_forest = None

        if not self.need_relpn:
            rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
            rels_to_gt = result.rels_to_gt

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        if self.need_relpn:
            # if self.mode == 'predcls':
            #    obj_logits = Variable(to_onehot(result.rm_obj_labels.data, self.num_classes))
            # else:
            #    obj_logits = result.rm_obj_dists
            obj_logits = result.rm_obj_dists
            roi_feat = obj_logits.detach()
            if self.relpn_embed_dim > 0:
                prob_feat = F.softmax(obj_logits, dim=1)
                relpn_embed_feat = prob_feat @ self.relpn_embed.weight
                roi_feat = torch.cat((roi_feat, relpn_embed_feat), 1)
            if self.relpn_with_bbox_info:
                sizes = Variable(torch.FloatTensor(im_sizes).cuda(im_inds.get_device()))
                sizes = sizes[im_inds]
                pos_feat = boxes.clone()
                pos_feat[:, 0::2] /= sizes[:, 1][:, None]
                pos_feat[:, 1::2] /= sizes[:, 0][:, None]
                roi_feat = torch.cat((roi_feat, pos_feat.detach()), 1)
            rel_proposal_inds, rel_proposal_scores, rel_anchor_labels, rel_anchor_scores = \
                self.relpn_head(rois, roi_feat, gt_boxes, gt_classes, gt_rels, result, image_offset, self.mode)
            result.rel_anchor_scores = rel_anchor_scores
            result.rel_anchor_labels = rel_anchor_labels
            result.rel_proposal_inds = rel_proposal_inds
            # print('before: **')
            # _cnt_proposal(rel_proposal_inds.data, gt_rels[:, :3].data, gt_classes[:, 0].data)
            if self.training:
                rel_proposal_labels, rels_to_gt = \
                    rel_proposal_target(rois.data, rel_proposal_inds.data, gt_boxes.data, gt_classes.data, gt_rels.data,
                                        image_offset, self.mode)
                result.rel_proposal_labels = rel_proposal_labels
                rel_inds = rel_proposal_labels[:, :3].data.clone()

                # print('after: **')
                # _cnt_proposal(rel_proposal_labels[:, :3].data, gt_rels[:, :3].data, gt_classes[:, 0].data)
                # compare the rel_proposal_labels with result.rel_labels
                # _compare(rel_proposal_labels.data, result.rel_labels.data)

            else:
                rel_inds = rel_proposal_inds.data.clone()
                # _compare(rel_proposal_inds.data, self.get_rel_inds(result.rel_labels, im_inds, boxes))

        ###!!!! need_relrank here!!!!

        # whole image feature, used for virtual node
        batch_size = result.fmap.shape[0]
        image_rois = Variable(torch.randn(batch_size, 5).fill_(0).cuda())
        for i in range(batch_size):
            image_rois[i, 0] = i
            image_rois[i, 1] = 0
            image_rois[i, 2] = 0
            image_rois[i, 3] = IM_SCALE
            image_rois[i, 4] = IM_SCALE
        image_fmap = self.obj_feature_map(result.fmap.detach(), image_rois)

        if self.mode != 'sgdet' and self.training:
            fg_rel_labels = result.rel_labels

        # Prevent gradients from flowing back into score_fc from elsewhere
        #'''
        result.rm_obj_dists, result.obj_preds, edge_ctx, forest = self.context(
            result.obj_fmap,
            result.rm_obj_dists.detach(),
            im_inds, result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
            boxes.data, result.boxes_all,
            arbitrary_forest,
            image_rois,
            image_fmap,
            fg_rel_labels if self.training else None,
            None,
            x,
            depth_maps=depth_maps if self.use_depth else None)
        #'''

        #rel_inds = self.get_hr_rel_inds(result.rel_labels, im_inds, boxes, forest)

        if self.need_relrank:
            assert sal_maps is not None
            if self.sal_input == 'empty':
                rel_union_rep = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
            else:
                if self.sal_input == 'sal':
                    sal_masks = sal_maps
                elif self.sal_input == 'area':
                    sal_masks = get_area_maps(result.fmap.shape[2:], im_inds.data, boxes.data, scale=16)
                    sal_masks = F.adaptive_avg_pool2d(sal_masks, result.fmap.shape[2:])
                else:
                    area_maps = get_area_maps(result.fmap.shape[2:], im_inds.data, boxes.data, scale=16)
                    area_maps = F.adaptive_avg_pool2d(area_maps, result.fmap.shape[2:])
                    sal_masks = (sal_maps + area_maps)
                rel_union_rep = self.visual_rep(result.fmap.detach() * sal_masks, rois, rel_inds[:, 1:])  # union box features
            union_boxes = union_regions(rois, rel_inds[:, 1:]).data
            pair_im_inds = rel_inds[:, 0]


            subj_boxes = boxes[rel_inds[:, 1]]
            obj_boxes = boxes[rel_inds[:, 2]]
            subj_xywh = center_size(subj_boxes)
            obj_xywh = center_size(obj_boxes)
            pos_rep = torch.cat([((obj_xywh[:, 0] - subj_xywh[:, 0])/torch.sqrt(subj_xywh[:, 2]*subj_xywh[:, 3]))[:, None],
                                 ((obj_xywh[:, 1] - subj_xywh[:, 1])/torch.sqrt(subj_xywh[:, 2]*subj_xywh[:, 3]))[:, None],
                                 torch.sqrt((obj_xywh[:, 2]*obj_xywh[:, 3])/(subj_xywh[:, 2]*subj_xywh[:, 3]))[:, None],
                                 (subj_xywh[:, 2]/subj_xywh[:, 3])[:, None],
                                 (obj_xywh[:, 2]/obj_xywh[:, 3])[:, None],
                                 bbox_overlaps_pair(subj_boxes, obj_boxes)[:, None]], 1)

            rel_rank_ctx = self.rank_context(rel_union_rep.detach(), pos_rep, pair_im_inds, union_boxes)
            result.rel_rank_scores = self.rank_visual_proj(rel_rank_ctx).view(-1).contiguous()


            if self.use_CE:
                result.rel_rank_dists = F.softmax(result.rel_rank_scores) # N, 2
            else:
                result.rel_rank_dists = F.sigmoid(result.rel_rank_scores) # N,


            if self.training:
                # add another column in rel_inds to locate the key rels
                # basic: use fg / bg as salient or not-salient: for vg200 dataset
                keylabel_byfgbg = Variable(torch.LongTensor(rel_inds.shape[0]).fill_(0).cuda())
                keylabel_byfgbg[torch.nonzero(rels_to_gt >= 0).view(-1)] = 1
                result.keylabel_byfgbg = torch.cat((Variable(rel_inds[:, 0][:, None]), keylabel_byfgbg[:, None]), 1)
                #loss = keyrel_loss(result.rel_rank_dists, result.keylabel_byfgbg, sample_num=512, margin=0.2)
                #loss.backward()
                #loss = FocalLoss(alpha=0.25, gamma=2)(result.rel_rank_scores, result.keylabel_byfgbg[:, 1].contiguous())
                #loss.backward()
                """
                pos_inds = torch.nonzero(result.keylabel_byfgbg[:, 1] == 1).view(-1)
                neg_inds = torch.nonzero(result.keylabel_byfgbg[:, 1] == 0).view(-1)
                print('****************************')
                #print('positive scores:', result.rel_rank_dists[pos_inds].view(-1))
                #print('negative scores:', result.rel_rank_dists[neg_inds].view(-1))
                print('positive min/max:', torch.min(result.rel_rank_dists[pos_inds].view(-1)),
                      torch.max(result.rel_rank_dists[pos_inds].view(-1)))
                print('negative min/max:', torch.min(result.rel_rank_dists[neg_inds].view(-1)),
                      torch.max(result.rel_rank_dists[neg_inds].view(-1)))
                print('margin:', torch.min(result.rel_rank_dists[pos_inds].view(-1)) - torch.max(result.rel_rank_dists[neg_inds].view(-1)))
                print('****************************')
                """
                # more: key_rels is not None,
                if key_rels is not None:
                    key_gt_rels = Variable(torch.LongTensor(gt_rels.shape[0]).fill_(0).cuda(result.rel_labels.get_device()))
                    offset = {}
                    for i, s, e in enumerate_by_image(gt_rels.data[:, 0]):
                        offset[i] = s
                    for i, s, e in enumerate_by_image(key_rels.data[:, 0]):
                        key_gt_rels[key_rels[s:e, 1] + offset[i]] = 1

                    keylabel_bykeyrel = rels_to_gt.clone()
                    keylabel_bykeyrel[torch.nonzero(rels_to_gt > -1).view(-1)] = key_gt_rels[
                        rels_to_gt[torch.nonzero(rels_to_gt > -1).view(-1)]]
                    # ignore the bg rels
                    if self.need_relpn:
                        keylabel_bykeyrel[torch.nonzero(result.rel_proposal_labels[:, 3] == 0).view(-1)] = -1
                    else:
                        keylabel_bykeyrel[torch.nonzero(result.rel_labels[:, 3] == 0).view(-1)] = -1
                    result.keylabel_bykeyrel = torch.cat((Variable(rel_inds[:, 0][:, None]), keylabel_bykeyrel[:, None]), 1)


        if edge_ctx is None:
            edge_rep = self.post_emb(result.obj_preds)
        else:
            edge_rep = self.post_lstm(edge_ctx)

        # Split into subject and object representations
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)

        subj_rep = edge_rep[:, 0]
        obj_rep = edge_rep[:, 1]

        prod_rep = torch.cat((subj_rep[rel_inds[:, 1]], obj_rep[rel_inds[:, 2]]), 1)
        prod_rep = self.post_cat(prod_rep)

        if self.use_encoded_box:
            # encode spatial info
            assert (boxes.shape[1] == 4)
            # encoded_boxes: [box_num, (x1,y1,x2,y2,cx,cy,w,h)]
            encoded_boxes = tree_utils.get_box_info(boxes)
            # encoded_boxes_pair: [batch_szie, (box1, box2, unionbox, intersectionbox)]
            encoded_boxes_pair = tree_utils.get_box_pair_info(encoded_boxes[rel_inds[:, 1]],
                                                              encoded_boxes[rel_inds[:, 2]])
            # encoded_spatial_rep
            spatial_rep = F.relu(self.encode_spatial_2(F.relu(self.encode_spatial_1(encoded_boxes_pair))))
            # element-wise multiply with prod_rep
            prod_rep = prod_rep * spatial_rep

        if self.use_vision:
            vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
            if self.limit_vision:
                # exact value TBD
                prod_rep = torch.cat((prod_rep[:, :2048] * vr[:, :2048], prod_rep[:, 2048:]), 1)
            else:
                prod_rep = prod_rep * vr

        if self.use_tanh:
            prod_rep = F.tanh(prod_rep)

        result.prod_rep = prod_rep

        result.rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                result.obj_preds[rel_inds[:, 1]],
                result.obj_preds[rel_inds[:, 2]],
            ), 1))

        #if self.training:
        #    return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        rel_rep = F.softmax(result.rel_dists, dim=1)

        if self.training:
            if self.need_relrank:
            # for rel-rank var2
                subj_scores = result.obj_scores[rel_inds[:, 1]]
                obj_scores = result.obj_scores[rel_inds[:, 2]]

                if self.use_dist:
                    mul_scores = result.rel_rank_dists
                else:
                    mul_scores = result.rel_rank_scores
                if self.has_grad:
                    result.rel_rank_scores = mul_scores * subj_scores * obj_scores * \
                                         rel_rep[:, 1:].max(1)[0]
                else:
                    result.rel_rank_scores = mul_scores * subj_scores.detach() * obj_scores.detach() * \
                                         rel_rep[:, 1:].detach().max(1)[0]
                return result
            else:
                return result

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        # get the processed results for captioning
        if self.need_caption:
            return filter_dets_for_caption(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds, rel_rep, prod_rep, image_fmap,
                           result.rel_rank_scores, seq_labels, mask_labels, coco_ids)
        elif self.gcn_caption:
            return filter_dets_for_gcn_caption(im_inds.data, result.obj_fmap, result.obj_scores, result.obj_preds, rel_inds,
                                               rel_rep, result.rel_rank_scores, seq_labels, mask_labels, coco_ids)
        else:
            return_im_inds = im_inds.data if self.dbname == 'vrd' else None
            return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[: , 1:], rel_rep, gt_boxes, gt_classes, gt_rels,
                           result.rel_rank_dists if self.use_dist else result.rel_rank_scores, forest,
                           return_forest=self.return_forest, im_inds=return_im_inds)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs


class RelPNHead(nn.Module):
    """relation proposal network"""

    def __init__(self, dim=512, num_classes=151, embed_dim=100, pos_dim=4):
        super(RelPNHead, self).__init__()
        roi_feat_dim = num_classes + embed_dim + pos_dim

        self.relpn_bilinear_sub = nn.Sequential(nn.Linear(roi_feat_dim, 64),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(64, 64))
        self.relpn_bilinear_obj = nn.Sequential(nn.Linear(roi_feat_dim, 64),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(64, 64))

    def forward(self, rois, roi_feat, gt_boxes, gt_classes, gt_rels, od_result, image_offset, mode):
        """

        :param rois: N, 5
        :param roi_feat: N, D
        :param gt_boxes:
        :return:
        """
        N, D = roi_feat.size(0), roi_feat.size(1)

        x_sub = self.relpn_bilinear_sub(roi_feat.detach())  # N, 64
        x_obj = self.relpn_bilinear_obj(roi_feat.detach())  # N, 64

        x_bilinear = torch.mm(x_sub, x_obj.permute(1, 0))  # N, N
        scores = F.sigmoid(x_bilinear)  # N, N

        rel_proposal_inds, rel_proposal_scores = rel_proposal(rois.data, scores.data,
                                                              nms_thresh=0.7,
                                                              pre_nms_topn=60000 if self.training else 6000,
                                                              post_nms_topn=10000 if self.training else 300)
        rel_anchor_labels = None
        rel_anchor_scores = None
        if self.training:
            if mode in ('predcls', 'sgcls'):
                rel_anchor_inds = od_result.rel_labels[:, :3].clone()  # (im_inds, ind1, ind2)
                anchor_labels = (od_result.rel_labels[:, -1, None] > 0).long()
                rel_anchor_labels = torch.cat((rel_anchor_inds, anchor_labels), 1)  # (im_inds, ind1, ind2, 0 or 1)
            else:  # sg_det
                rel_anchor_labels = rel_anchor_target(rois.data, gt_boxes.data, gt_classes.data, scores.data,
                                                      gt_rels.data, image_offset)

            rel_anchor_scores = scores[rel_anchor_labels[:, 1], rel_anchor_labels[:, 2]]

        return rel_proposal_inds, rel_proposal_scores, rel_anchor_labels, rel_anchor_scores



def _cnt_proposal(proposal_inds, gt_inds, im_inds):
    proposal_inds_np = proposal_inds.cpu().numpy()
    offset = {}
    for i, s, e in enumerate_by_image(im_inds):
        offset[i] = s
    for i, s, e in enumerate_by_image(gt_inds[:, 0]):
        gt_inds[s:e, 1:3] += offset[i]
    gt_inds_np = gt_inds.cpu().numpy()
    res = intersect_2d(proposal_inds_np, gt_inds_np)
    num_correct = len(np.where(np.sum(res, 1))[0])
    num_gt = gt_inds_np.shape[0]
    num_proposal = proposal_inds_np.shape[0]
    print(num_correct, num_gt, num_proposal, num_correct / num_gt, num_correct / num_proposal)


def _compare(rel_proposal_labels, result_rel_labels):
    proposal_labels_np = rel_proposal_labels.cpu().numpy()
    result_labels_np = result_rel_labels.cpu().numpy()
    print(proposal_labels_np.shape[0], result_labels_np.shape[0])

    res = intersect_2d(proposal_labels_np, result_labels_np)
    # res_inds = intersect_2d(proposal_labels_np[:, :3], result_labels_np[:, :3])
    num_res_cor = len(np.where(np.sum(res, 1))[0])
    # num_res_inds_cor = len(np.where(np.sum(res_inds, 1))[0])
    # print(num_res_cor, num_res_inds_cor)
