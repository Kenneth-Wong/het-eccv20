import torch
import numpy as np
from .._ext import co_nms

def apply_co_nms(pair_boxes, nms_thresh, post_nms_topn):
    """

    :param pair_boxes: Npair, 8, sorted pair boxes for a single image
    :param nms_thresh:
    :return:
    Note - this function is non-differentiable so everything is assumed to be a tensor, not
    a variable.
    """
    keep = torch.IntTensor(pair_boxes.size(0))
    num_out = co_nms.co_nms_apply(keep, pair_boxes, nms_thresh)
    num_out = min(num_out, post_nms_topn)
    keep = keep[:num_out].long()
    keep = keep.cuda(pair_boxes.get_device())
    return keep, num_out