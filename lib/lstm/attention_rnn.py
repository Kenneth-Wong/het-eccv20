import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from typing import Optional, Tuple

from lib.fpn.box_utils import nms_overlaps
from lib.word_vectors import obj_edge_vectors
from .highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
import numpy as np
from lib.pytorch_misc import enumerate_by_image, transpose_packed_sequence_inds


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
    num_im = int(im_inds[-1] + 1)
    rois_per_image = scores.new(num_im)
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    # get the sorted batch order, the image with more rois ranks first
    batch_perm = torch.LongTensor(np.argsort(np.array(lengths))[::-1].copy()).cuda(im_inds.get_device())
    _, batch_inv_perm = torch.sort(batch_perm)
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

    return perm, inv_perm, batch_perm, batch_inv_perm, ls_transposed

def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.autograd.Variable):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def pack_vectors(im_inds, vec_inps):
    num_im = int(im_inds[-1] + 1)
    max_num_roi = 0
    im2roi = []
    ls_rois = []
    for i, s, e in enumerate_by_image(im_inds):
        im2roi.append((s, e))
        max_num_roi = max(max_num_roi, e - s)
        ls_rois.append(e - s)
    packed_tensor = Variable(torch.FloatTensor(num_im, max_num_roi, vec_inps.shape[1]).fill_(0).cuda(vec_inps.get_device()))
    for i, seg in enumerate(im2roi):
        packed_tensor[i, :ls_rois[i]] = vec_inps[seg[0]:seg[1], :]
    return packed_tensor, np.array(ls_rois)

def unpack_to_tensors(att_list, ls_rois):
    """

    :param att_list: a list, each element is a B*N tensor, B is the batch size of current tiemstep
    :return:
    """
    num_im = att_list[0].shape[0]
    atten_per_imgs = [[] for _ in range(num_im)]
    for att_tensor in att_list:
        ls_batch = att_tensor.shape[0]
        ls_roi = ls_rois[:ls_batch]
        for b in range(att_tensor.shape[0]):
            atten_per_imgs[b].append(att_tensor[b, :ls_roi[b]].view(1, -1))
    atten_per_imgs = [torch.cat(a, 0) for a in atten_per_imgs]
    return atten_per_imgs


class AttentionRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, visual_dim, attention_dim, recurrent_dropout_probability=0.2,
                 use_highway=True, use_input_projection_bias=True, use_CE=False):
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        """
        super(AttentionRNN, self).__init__()

        self.hidden_size = hidden_dim
        self.inputs_dim = input_dim
        self.visual_dim = visual_dim
        self.attention_dim = attention_dim

        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.use_highway = use_highway

        self.initial_state_linearity = torch.nn.Linear(visual_dim, self.hidden_size)
        self.initial_cell_linearity = torch.nn.Linear(visual_dim, self.hidden_size)

        self.visual_attention_encode = torch.nn.Linear(visual_dim, attention_dim)
        self.hidden_attention_encode = torch.nn.Linear(self.hidden_size, attention_dim)
        self.att_proj = torch.nn.Linear(attention_dim, 1)

        self.use_CE = use_CE

        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        if use_highway:
            self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size,
                                                   bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size,
                                                   bias=True)
            self.atten_linearity = torch.nn.Linear(self.visual_dim, 5 * self.hidden_size,
                                                   bias=True)
        else:
            self.input_linearity = torch.nn.Linear(self.input_size, 4 * self.hidden_size,
                                                   bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(self.hidden_size, 4 * self.hidden_size,
                                                   bias=True)
            self.atten_linearity = torch.nn.Linear(self.visual_dim, 4 * self.hidden_size,
                                                   bias=True)

        out_unit = 2 if self.use_CE else 1
        self.out = torch.nn.Linear(self.hidden_size, out_unit)

        # self.out = nn.Linear(self.hidden_size, len(self.classes))
        self.reset_parameters()

    def sort_rois(self, batch_idx, confidence, box_priors):
        scores = confidence
        return _sort_by_score(batch_idx, scores)

    @property
    def input_size(self):
        return self.inputs_dim

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])
        block_orthogonal(self.atten_linearity.weight.data, [self.hidden_size, self.visual_dim])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)
        self.atten_linearity.bias.data.fill_(0.0)
        self.atten_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, attention_input, dropout_mask=None):
        """
        Does the hairy LSTM math
        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)
        projected_atten = self.atten_linearity(attention_input)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_atten[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_atten[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_atten[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_atten[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_atten[:, 4 * self.hidden_size:5 * self.hidden_size])
            highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory


    def fatt_function(self, visual_inputs, hidden_state, ls_rois):
        """

        :param visual_inputs: B, N, V
        :param hidden_state: B, D
        :param ls_rois: number of effective rois of each image
        :return: attention value, B, 1, N
        """
        batch_size, N = visual_inputs.size(0), visual_inputs.size(1)
        h_att = F.relu(self.visual_attention_encode(visual_inputs.view(-1, self.visual_dim)).view(batch_size, N, -1) + \
                    self.hidden_attention_encode(hidden_state).unsqueeze(1))  # B, N, att_dim
        out_att = self.att_proj(h_att.view(-1, self.attention_dim)).view(batch_size, N)  # B, N
        for b, ls in enumerate(ls_rois[:batch_size]):
            out_att[b, :ls] = F.softmax(out_att[b, :ls].clone())
            if ls < N:
                out_att[b, ls:] = 0
        context = torch.bmm(out_att.unsqueeze(1), visual_inputs).squeeze(1) # B, V
        return context, out_att

    def rnn_forward(self,  # pylint: disable=arguments-differ
                    inputs: PackedSequence,
                    visual_inputs: torch.Tensor,
                    ls_rois: np.array,
                    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Parameters
        ----------
        inputs : PackedSequence, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.

        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        Returns
        -------
        A PackedSequence containing a torch.FloatTensor of shape
        (batch_size, num_timesteps, output_dimension) representing
        the outputs of the LSTM per timestep and a tuple containing
        the LSTM state, with shape (1, batch_size, hidden_size) to
        match the Pytorch API.
        """
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))

        assert isinstance(inputs, PackedSequence)
        sequence_tensor, batch_lengths = inputs
        batch_size = batch_lengths[0]

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = Variable(sequence_tensor.data.new()
                                       .resize_(batch_size, self.hidden_size).fill_(0))
            previous_state = Variable(sequence_tensor.data.new()
                                      .resize_(batch_size, self.hidden_size).fill_(0))
        else:
            assert len(initial_state) == 2
            previous_state = initial_state[0]
            previous_memory = initial_state[1]

        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, previous_memory)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        out_alpha = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = sequence_tensor[start_ind:end_ind]  # each batch's input

            # visual_inputs[:l_batch] : B_t, N, Dim
            # previous_state: B_t, Dim
            # return: attention value, B_t, 1, N, weighted sum
            attention_input, alpha = self.fatt_function(visual_inputs[:l_batch], previous_state, ls_rois)

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, attention_input,
                                                                  dropout_mask=dropout_mask)

            pred_dist = self.out(previous_state)  # B_t, 2 or B_t, 1
            out_dists.append(pred_dist)
            out_alpha.append(alpha)
        return torch.cat(out_dists, 0), out_alpha

    def forward(self, sal_inps, visual_rep, obj_boxes, union_boxes, im_centers):
        boxes_center = torch.cat(((union_boxes[:, 4] + union_boxes[:, 2])[:, None] / 2,
                                  (union_boxes[:, 3] + union_boxes[:, 1])[:, None] / 2), 1)
        confidence = torch.sum((boxes_center - im_centers) ** 2, 1)
        confidence = (confidence * -1 + confidence.max()) / (confidence.max() + 1)  # closed to image center is higher
        perm, inv_perm, batch_perm, batch_inv_perm, ls_transposed = self.sort_rois(union_boxes[:, 0].long(), confidence, union_boxes[:, 1:])

        sal_inp_rep = sal_inps[perm].contiguous()  # batch order is re ordered
        input_packed = PackedSequence(sal_inp_rep, ls_transposed)
        rel_visual_packed, ls_rois = pack_vectors(obj_boxes[:, 0].long(), visual_rep)  # B, N, D
        rel_visual_packed = rel_visual_packed[batch_perm]  # reorder according to batch
        ls_rois = ls_rois[batch_perm.cpu().numpy()]

        # build inital hidden state and cell
        mean_state = torch.sum(rel_visual_packed, 1) / Variable(torch.FloatTensor(ls_rois).cuda(
            rel_visual_packed.get_device()).view(-1, 1))  # B, D
        initial_state = self.initial_state_linearity(mean_state)  # B, H
        initial_cell = self.initial_cell_linearity(mean_state)  # B, H
        out_scores, att_alpha = self.rnn_forward(input_packed, rel_visual_packed, ls_rois, (initial_state, initial_cell))
        out_scores = out_scores[inv_perm]
        if out_scores.size(1) == 1:
            out_scores = out_scores.view(-1)
        att_alpha = unpack_to_tensors(att_alpha, ls_rois)
        att_alpha = [att_alpha[ind] for ind in batch_inv_perm.cpu().numpy()]
        ls_rois = ls_rois[batch_inv_perm.cpu().numpy()]
        return out_scores, att_alpha


