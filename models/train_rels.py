"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

from dataloaders.visual_genome import VGDataLoader, VG
from dataloaders.visual_genome200 import VG200
from dataloaders.visual_genome200 import VGDataLoader as VGDataLoader200
from dataloaders.visual_genome200_keyrel import VG200_Keyrel
from dataloaders.visual_genome200_keyrel import VGDataLoader as VGDataLoader200_KR
from dataloaders.visual_genome200_keyrel_captions import VG200_Keyrel_captions
from dataloaders.visual_genome200_keyrel_captions import VGDataLoader as VGDataLoader200_KR_cap
import numpy as np
from torch import optim
import torch
from torch.autograd import Variable
import pandas as pd
import time
import os

import torch.nn as nn
from config import ModelConfig, BOX_SCALE, IM_SCALE, FREQ_WEIGHT, SAMPLE_NUM
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, keyrel_loss, FocalLoss
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

conf = ModelConfig()
if conf.model == 'het':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

assert (not (conf.vg200 and conf.vg200_kr))
VGdata = VG
VGLoader = VGDataLoader
dbname = 'vg'
if conf.vg200:
    VGdata = VG200
    VGLoader = VGDataLoader200
    dbname = 'vg200'
if conf.vg200_kr:
    VGdata = VG200_Keyrel
    VGLoader = VGDataLoader200_KR
    dbname = 'vg200_kr'
if conf.vg200_kr_cap:
    VGdata = VG200_Keyrel_captions
    VGLoader = VGDataLoader200_KR_cap
    dbname = 'vg200_kr_cap'


train, val, test = VGdata.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                                 use_proposals=conf.use_proposals,
                                 filter_non_overlap=conf.mode == 'sgdet')

train_loader, val_loader = VGLoader.splits(train, val, mode='rel',
                                           batch_size=conf.batch_size,
                                           num_workers=conf.num_workers,
                                           num_gpus=conf.num_gpus)

_, test_loader = VGLoader.splits(train, test, mode='rel',
                                 batch_size=conf.batch_size,
                                 num_workers=conf.num_workers,
                                 num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order, pick_parent=conf.pick_parent,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals, isc_thresh=conf.isc_thresh,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    use_encoded_box=conf.use_encoded_box,
                    draw_tree=conf.draw_tree,
                    limit_vision=conf.limit_vision,
                    need_relpn=conf.relpn,
                    need_relrank=conf.relrank,
                    use_CE=conf.use_CE,
                    dbname=dbname,
                    sal_input=conf.sal_input,
                    use_depth=conf.use_depth,
                    has_grad=conf.has_grad,
                    use_dist=conf.use_dist,
                    return_forest=conf.test_forest
                    )

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False


def fix_batchnorm(model):
    if isinstance(model, list):
        for m in model:
            fix_batchnorm(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                # print('Fix BatchNorm1d')
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                # print('Fix BatchNorm2d')
                m.eval()
            elif isinstance(m, nn.BatchNorm3d):
                # print('Fix BatchNorm3d')
                m.eval()
            elif isinstance(m, nn.Dropout):
                # print('Fix Dropout')
                m.eval()
            elif isinstance(m, nn.AlphaDropout):
                # print('Fix AlphaDropout')
                m.eval()


def fix_rest_net(model):
    for n, param in model.named_parameters():
        param.requires_grad = False
    for n, param in model.context.feat_preprocess_net.named_parameters():
        param.requires_grad = True
    for n, param in model.context.rl_sub.named_parameters():
        param.requires_grad = True
    for n, param in model.context.rl_obj.named_parameters():
        param.requires_grad = True
    for n, param in model.context.rl_scores.named_parameters():
        param.requires_grad = True
    # fix batchnorm during self critic
    fix_batchnorm(model)


print(print_para(detector), flush=True)


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n, p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n, p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    #att_rnn_params = [p for n, p in detector.named_parameters() if n.startswith('attention_rnn') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adadelta(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler


ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])

    detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
    '''
    if conf.relrank:
        detector.rankLSTM_roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.rankLSTM_roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.rankLSTM_roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.rankLSTM_roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
    '''
detector.cuda()


def train_epoch(epoch_num):
    detector.train()

    # two evaluator to calculate recall
    base_evaluator = BasicSceneGraphEvaluator.all_modes(num_predicates=80 if conf.vg200 or conf.vg200_kr or conf.vg200_kr_cap else 50)
    train_evaluator = BasicSceneGraphEvaluator.all_modes(num_predicates=80 if conf.vg200 or conf.vg200_kr or conf.vg200_kr_cap else 50)

    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        if batch[0][4].shape[0] > 90:
            # print('lager: ', batch[0][4].shape[0])
            continue

        batch_train = train_batch
        tr.append(batch_train(b, batch, verbose=b % (conf.print_interval * 10) == 0, base_evaluator=base_evaluator,
                              train_evaluator=train_evaluator))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(count, b, verbose=False, base_evaluator=None, train_evaluator=None):
    result = detector[b]

    losses = {}
    losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    if conf.relpn:
        losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_proposal_labels[:, -1])
    else:
        losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])


    if conf.relpn:
        losses['rel_proposal_loss'] = F.binary_cross_entropy(result.rel_anchor_scores, result.rel_anchor_labels[:, -1].float())

    if conf.relrank:
        losses['keyrel_fgbg_loss'] = keyrel_loss(result.rel_rank_scores, result.keylabel_byfgbg, margin=conf.margin1)
        if conf.vg200_kr:
            losses['keyrel_sal_loss'] = keyrel_loss(result.rel_rank_scores, result.keylabel_bykeyrel, margin=conf.margin1)

    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes(num_predicates=80 if conf.vg200 or conf.vg200_kr or conf.vg200_kr_cap else 50)
    for val_b, batch in enumerate(val_loader):
        if val_b < conf.val_size:
            val_batch(conf.num_gpus * val_b, batch, evaluator)
        else:
            break
    evaluator[conf.mode].print_stats()
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def test_recall():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes(num_predicates=80 if conf.vg200 or conf.vg200_kr else 50)
    for val_b, batch in enumerate(test_loader):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats()
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, rel_rank_scores_i, gt_boxes, gt_classes, gt_rels, forest) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
            'gt_key_rels': val.key_rel_idxes[batch_num + i].copy() if conf.vg200_kr else None
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
            'rel_rank_scores': rel_rank_scores_i,
            'forest': forest
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


print("Training starts now!")
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

best_recall = -1.
best_epoch = -1


for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):

    lr_set = set([pg['lr'] for pg in optimizer.param_groups])
    print('lr set: ', lr_set)
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    mAp = val_epoch()
    scheduler.step(mAp)

    if mAp > best_recall:
        best_recall = mAp
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
        }, os.path.join(conf.save_dir, '{}-best.tar'.format('vgrel')))
    print('**** best epoch: {}, \t best mAp: {:.4f} *****'.format(best_epoch, best_recall))
    
    if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size) / 99.0 for pg in optimizer.param_groups]):
        print("exiting training early", flush=True)
        break
