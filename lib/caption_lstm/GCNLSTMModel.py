# ---------------------------------------------------------------
# GCNLSTMModel.py
# Set-up time: 2020/2/17 下午9:35
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from lib.pytorch_misc import enumerate_by_image
from lib.word_vectors import obj_edge_vectors
import random
import numpy as np

from dataloaders.visual_genome200_keyrel_captions import VG200_Keyrel_captions
from dataloaders.visual_genome200 import VG200
from lib.get_dataset_counts import get_counts


def _construct_graph(rels, obj_classes, pred_classes, num_relation=-1, freq_matrix=None):
    rel_im_inds = rels[:, 0]
    rels_all = []
    pred_classes_all = []
    for i, s, e in enumerate_by_image(rel_im_inds):
        rels_i = rels[s:e, :]
        pred_classes_i = pred_classes[s:e]

        subj_categories = obj_classes[rels_i[:, 1]][:, None]
        obj_categories = obj_classes[rels_i[:, 2]][:, None]
        categories_info = torch.cat((subj_categories, obj_categories, pred_classes_i), 1)

        # compute freqency baseline: reranking based on triplet frequency
        if freq_matrix is not None:
            categories_info_np = categories_info.cpu().numpy()
            freqs = []
            for cat in categories_info_np:
                freqs.append(freq_matrix[cat[0], cat[1], cat[2]])
            sort_index = torch.from_numpy(np.argsort(np.array(freqs) * -1)).cuda(rels.get_device())
            rels_i = rels_i[sort_index, :]
            pred_classes_i = pred_classes_i[sort_index]
            categories_info = categories_info[sort_index, :]

        this_num_rel = rels_i.shape[0]
        # no constraint
        if num_relation <= 0:
            pass
        elif num_relation <= this_num_rel:
            rels_i = rels_i[:num_relation, :]
            pred_classes_i = pred_classes_i[:num_relation]
            categories_info = categories_info[:num_relation, :]
        else:
            # oversample
            sample_inds = torch.from_numpy(
                np.random.choice(np.arange(this_num_rel, dtype=np.int32), num_relation, replace=True)).long().cuda(
                rels.get_device())
            rels_i = rels_i[sample_inds, :]
            pred_classes_i = pred_classes_i[sample_inds]
            categories_info = categories_info[sample_inds, :]
        rels_all.append(rels_i)
        pred_classes_all.append(pred_classes_i)
    return torch.cat(rels_all, 0), torch.cat(pred_classes_all, 0)


def _PadVisual(im_inds, region_feats, seq_per_img):
    """
    """
    D = region_feats.size(-1)
    max_region_num = -1
    for i, s, e in enumerate_by_image(im_inds):
        max_region_num = max(max_region_num, e - s)

    region_feats_enlarged = []
    effective_regions = []
    for i, s, e in enumerate_by_image(im_inds):
        num_region = e - s
        effective_regions += [num_region] * seq_per_img

        region_feats_i = region_feats[s:e, :]
        pad_num = max_region_num -  num_region
        if pad_num > 0:
            zero_padding = Variable(
                torch.zeros(seq_per_img, pad_num, D).float().cuda(region_feats.get_device()))
            # 5 * Nmax * D
            padded_region_feats = torch.cat([torch.cat([region_feats_i[None, :, :]] * seq_per_img, 0), zero_padding], 1)
        else:
            padded_region_feats = torch.cat([region_feats_i[None, :, :]] * seq_per_img, 0)

        region_feats_enlarged.append(padded_region_feats)

    region_feats_enlarged = torch.cat(region_feats_enlarged, 0)  # 5B * Nmax * D

    return region_feats_enlarged, effective_regions


class RelCaptionCore(nn.Module):
    def __init__(self, input_encoding_size, rnn_type='lstm', rnn_size=512, num_layers=1, drop_prob_lm=0.5,
                 fc_feat_size=4096, att_feat_size=512):
        super(RelCaptionCore, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.drop_prob_lm = drop_prob_lm
        self.fc_feat_size = fc_feat_size
        self.att_feat_size = att_feat_size

        self.rnn_1 = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size + self.rnn_size,
                                                        self.rnn_size, self.num_layers, bias=False,
                                                        dropout=self.drop_prob_lm)
        self.rnn_2 = getattr(nn, self.rnn_type.upper())(self.fc_feat_size + self.rnn_size, self.rnn_size,
                                                        self.num_layers, bias=False,
                                                        dropout=self.drop_prob_lm)
        self.ctx2att = nn.Linear(self.fc_feat_size, self.att_feat_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_feat_size)
        self.att_net = nn.Linear(self.att_feat_size, 1)

    def build_mask(self, bs, max_num, num_region, region_feats):
        mask = Variable(torch.zeros(bs, max_num).cuda(region_feats.get_device()))
        for b, n in enumerate(num_region):
            mask[b, :n] = 1
        return mask

    def forward(self, xt, mean_region_feats, region_feats, num_region, state):
        bs = region_feats.size(0)
        mask = self.build_mask(bs, region_feats.size(1), num_region, region_feats)
        rnn1_state = (state[0], state[1])
        rnn2_state = (state[2], state[3])
        input_h = rnn2_state[0].squeeze(0)
        out_1, rnn1_state = self.rnn_1(torch.cat([input_h, xt, mean_region_feats], 1).unsqueeze(0),
                                       rnn1_state)  # 1 x batch x rnn_size
        h_1, _ = rnn1_state
        s_h_1 = h_1.squeeze(0)  # batch x rnn_size

        attention = F.tanh(self.ctx2att(region_feats) + self.h2att(s_h_1).unsqueeze(1))  # batch x Nr x att_feat_size
        attention = self.att_net(attention).squeeze(2)  # batch x Nr
        attention = torch.exp(attention - torch.max(attention, dim=1, keepdim=True)[0]) * mask
        attention = attention / torch.sum(attention, dim=1, keepdim=True)
        attention = attention.unsqueeze(1)  # batch x 1 x Nr
        att_input = torch.bmm(attention, region_feats).squeeze(1)  # batch x fc_feat_size
        out_2, rnn2_state = self.rnn_2(torch.cat([att_input, s_h_1], 1).unsqueeze(0),
                                       rnn2_state)
        return out_2.squeeze(0), rnn1_state + rnn2_state


class GCNLSTMModel(nn.Module):
    def __init__(self, vocabs, vocab_size, input_encoding_size, Dconv=4096, num_predicate=81, rnn_type='lstm',
                 rnn_size=512, num_layers=1, drop_prob_lm=0.5, seq_length=16, seq_per_img=5, att_feat_size=512,
                 num_relation=20, freq_bl=False):
        super(GCNLSTMModel, self).__init__()

        # params for GCN
        self.Dconv = Dconv
        self.W_conv = nn.Parameter(torch.zeros(self.Dconv, self.Dconv * 3))
        self.b_lab = nn.Parameter(torch.zeros(num_predicate, self.Dconv))  # line 0 for self loop
        self.W_g = nn.Parameter(torch.zeros(self.Dconv, 3))
        self.b_glab = nn.Parameter(torch.zeros(num_predicate, 1))

        # params and settings for language
        self.vocabs = vocabs
        self.vocabs['0'] = '__SENTSIGN__'  ## ix
        self.vocabs = {i: self.vocabs[str(i)] for i in range(len(self.vocabs))}
        vocab_list = [self.vocabs[i] for i in range(len(self.vocabs))]
        self.vocab_size = vocab_size + 1  # including all the words and <UNK>, and 0 for <start>/<end>

        self.input_encoding_size = input_encoding_size
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.drop_prob_lm = drop_prob_lm
        self.seq_length = seq_length
        self.fc_feat_size = Dconv
        self.ss_prob = 0.0  # Schedule sampling probability
        self.num_relation_per_img = num_relation
        self.seq_per_img = seq_per_img

        self.freq_bl = freq_bl

        embed_vec = obj_edge_vectors(vocab_list, wv_dim=self.input_encoding_size)
        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.embed.weight.data = embed_vec.clone()

        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.core = RelCaptionCore(input_encoding_size, rnn_type, rnn_size, num_layers, drop_prob_lm,
                                   self.fc_feat_size, att_feat_size)

        if self.freq_bl:
            self.freq_matrix, _ = get_counts(
                train_data=VG200(mode='train', filter_duplicate_rels=False, num_val_im=1000), must_overlap=True)
        else:
            self.freq_matrix = None

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
        for i in range(3):
            self.W_conv.data[:, i * self.Dconv : (i + 1) * self.Dconv].uniform_(-initrange, initrange)
            self.W_g.data[:, i : (i + 1)].uniform_(-initrange, initrange)

    def conv_table(self, num_region, rels, predicates):
        # 3 column: neighbour_id, subject(0)/object(1)/selfLoop(2), predicate_id
        neighbour_table = {i: [] for i in range(num_region)}
        for rel_id, rel in enumerate(rels):
            subj, obj = rel[1], rel[2]
            neighbour_table[subj].append([obj, 0, predicates[rel_id]])
            neighbour_table[obj].append([subj, 1, predicates[rel_id]])
        # fill the self loop
        for i in range(num_region):
            neighbour_table[i].append([i, 2, 0])  # 0 for self loop in b_lab
            neighbour_table[i] = Variable(torch.from_numpy(np.array(neighbour_table[i])).long().view(-1, 3).cuda(rels.get_device()))

        return neighbour_table

    def GCN(self, region_feats, pred_classes, rels):
        """
        Compute graph convolution and generate captions with attention-LSTM
        region_feats: (N, Dconv)
        im_inds: Tensor, not Variable
        :return:
        """
        num_region = region_feats.shape[0]
        if region_feats.dim() != 2:
            region_feats = region_feats.view(num_region, -1)  # N, Dconv

        neighbor_table = self.conv_table(num_region, rels, pred_classes)

        neighbor_feats = torch.mm(region_feats, self.W_conv).view(-1, self.Dconv)  # 3N, Dconv
        gate_feats = torch.mm(region_feats, self.W_g).view(-1, 1)  # 3N, 1

        upd_region_feats = []
        for i in range(num_region):
            neighbor_info = neighbor_table[i]
            neighbor_ids = neighbor_info[:, 0]
            edge_type = neighbor_info[:, 1]
            predicate_ids = neighbor_info[:, 2]

            gate = F.sigmoid(torch.index_select(gate_feats, 0, neighbor_ids * 3 + edge_type) + \
                             torch.index_select(self.b_glab, 0, predicate_ids))  # Nb * 1

            upd_v = torch.sum(gate * (
                    torch.index_select(neighbor_feats, 0, neighbor_ids * 3 + edge_type) + torch.index_select(
                self.b_lab, 0, predicate_ids)), 0).view(1, -1)
            upd_v = F.relu(upd_v)
            upd_region_feats.append(upd_v)
        upd_region_feats = torch.cat(upd_region_feats, 0)
        return upd_region_feats

    def forward(self, im_inds, region_feats, pred_classes, rels, obj_classes, seq_labels, mask_labels):

        """
        1. Sample the edges and construct the graph for GCN.
        """
        rels, pred_classes = _construct_graph(rels, obj_classes, pred_classes, self.num_relation_per_img,
                                              self.freq_matrix)
        region_feats = self.GCN(region_feats, pred_classes, rels)

        """
        2. Enlarge region features for captioning.
        """
        # (5B) * Nmax * 4096
        region_feats, num_region = _PadVisual(im_inds, region_feats, self.seq_per_img)
        mean_region_feats = []
        for b in range(len(region_feats)):
            mean_region_feats.append(torch.mean(region_feats[b, :num_region[b]], dim=0, keepdim=True))
        mean_region_feats = torch.cat(mean_region_feats, 0)  # (5B) * 4096

        rnn1_state = (
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
        rnn2_state = (
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
        state = rnn1_state + rnn2_state

        bs = region_feats.size(0)

        outputs = []
        for i in range(seq_labels.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = region_feats.data.new(bs).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq_labels[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq_labels[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq_labels[:, i].clone()
            if i >= 1 and seq_labels[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, mean_region_feats, region_feats, num_region, state)
            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)  # b x T x vocab_size

    def get_logprobs_state(self, it, tmp_mean_region_feats, tmp_region_feats, tmp_num_region, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_mean_region_feats, tmp_region_feats, tmp_num_region, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return logprobs, state

    def sample(self, im_inds, region_feats, pred_classes, rels, obj_classes, beam_size=3, sample_max=1,
               temperature=1.0):
        if beam_size > 1:
            return self.sample_beam(im_inds, region_feats, pred_classes, rels, obj_classes, beam_size)

        rels, pred_classes = _construct_graph(rels, obj_classes, pred_classes, self.num_relation_per_img,
                                              self.freq_matrix)
        region_feats = self.GCN(region_feats, pred_classes, rels)

        # Pad data, now each image only have one copy instead of seq_per_image
        # (B) * Nmax * 4096
        region_feats, num_region = _PadVisual(im_inds, region_feats, seq_per_img=1)
        mean_region_feats = []
        for b in range(len(region_feats)):
            mean_region_feats.append(torch.mean(region_feats[b, :num_region[b]], dim=0, keepdim=True))
        mean_region_feats = torch.cat(mean_region_feats, 0)  # (B) * 4096

        rnn1_state = (
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
        rnn2_state = (
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
            Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
        state = rnn1_state + rnn2_state

        bs = region_feats.size(0)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <start>
                it = region_feats.data.new(bs).long().zero_()  # (batch, )
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)  # (batch, )
                it = it.view(-1).long()  # (batch, )
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                it = it.view(-1).long()
            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, mean_region_feats, region_feats, num_region, state)

            logprobs = F.log_softmax(self.logit(self.dropout(output)))
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)  # B x T

    def sample_beam(self, im_inds, region_feats, pred_classes, rels, obj_classes, beam_size):

        rels, pred_classes = _construct_graph(rels, obj_classes, pred_classes, self.num_relation_per_img,
                                              self.freq_matrix)
        region_feats = self.GCN(region_feats, pred_classes, rels)

        # Pad data, now each image only have one copy instead of seq_per_image
        # (B) * Nmax * 4096
        region_feats, num_region = _PadVisual(im_inds, region_feats, seq_per_img=1)
        mean_region_feats = []
        for b in range(len(region_feats)):
            mean_region_feats.append(torch.mean(region_feats[b, :num_region[b]], dim=0, keepdim=True))
        mean_region_feats = torch.cat(mean_region_feats, 0)  # (B) * 4096

        #rnn1_state = (
        #    Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
        #    Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
        #rnn2_state = (
        #    Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
        #    Variable(torch.randn(region_feats.size(0), self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
        #state = rnn1_state + rnn2_state
        bs = region_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a ' \
                                                 'few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, bs).zero_()  # seq_length x bs
        seqLogprobs = torch.FloatTensor(self.seq_length, bs)

        # lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(bs)]
        for k in range(bs):
            tmp_region_feats = region_feats[k:k + 1].expand(*((beam_size,) + region_feats.size()[1:])).contiguous()
            tmp_mean_region_feats = mean_region_feats[k:k + 1].expand(beam_size, mean_region_feats.size(1))
            tmp_num_region = [num_region[k]] * beam_size
            rnn1_state = (
                Variable(torch.randn(beam_size, self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
                Variable(torch.randn(beam_size, self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
            rnn2_state = (
                Variable(torch.randn(beam_size, self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0),
                Variable(torch.randn(beam_size, self.rnn_size).cuda(region_feats.get_device())).unsqueeze(0))
            tmp_state = rnn1_state + rnn2_state
            #tmp_state = tuple([s.squeeze(0)[k:k+1].expand(beam_size, self.rnn_size).unsqueeze(0).contiguous() for s in state])

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
            done_beams = []
            for t in range(1):
                if t == 0:  # input <bos>
                    it = region_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                output, tmp_state = self.core(xt, tmp_mean_region_feats, tmp_region_feats,
                                          tmp_num_region, tmp_state)
                logprobs = F.log_softmax(self.logit(self.dropout(output)))
            self.done_beams[k] = self.beam_search(tmp_state, logprobs, tmp_mean_region_feats,
                                                  tmp_region_feats, tmp_num_region, beam_size=beam_size)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def beam_search(self, state, logprobs, *args, **kwargs):
        """
        Note: "state" contains all variables that change as times goes by, while those in args are fixed.
        """

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append(dict(c=ix[q, c], q=q,
                                           p=candidate_logprob,
                                           r=local_logprob))
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = tuple([s.clone() for s in state])
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        beam_size = kwargs['beam_size']
        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        # running sum of logprobs for each beam
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            state, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)
            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000
            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(Variable(it.cuda()), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams
