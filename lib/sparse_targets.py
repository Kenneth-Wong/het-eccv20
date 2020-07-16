from lib.word_vectors import obj_edge_vectors
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from config import DATA_PATH
import os
from lib.get_dataset_counts import get_counts

from dataloaders.visual_genome import VG
from dataloaders.visual_genome200 import VG200
from dataloaders.visual_genome200_keyrel import VG200_Keyrel
from dataloaders.visual_genome200_keyrel_captions import VG200_Keyrel_captions
from dataloaders.vrd import VRD

class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, dbname='vg', eps=1e-3):
        super(FrequencyBias, self).__init__()
        if dbname == 'vg':
            db = VG
        elif dbname == 'vg200':
            db = VG200
        elif dbname == 'vg200_kr':
            db = VG200_Keyrel
        elif dbname == 'vg200_kr_cap':
            db = VG200_Keyrel_captions
        elif dbname == 'vrd':
            db = VRD

        fg_matrix, bg_matrix = get_counts(train_data=db(mode='train', filter_duplicate_rels=False), must_overlap=True)
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix

        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])

        self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.obj_baseline.weight.data = pred_dist

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline
