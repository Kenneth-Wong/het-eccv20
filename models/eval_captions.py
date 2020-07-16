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
import json
import glob
from data.transform_annotations import transform_annos

import torch.nn as nn
from config import ModelConfig, BOX_SCALE, IM_SCALE, FREQ_WEIGHT, SAMPLE_NUM
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm, keyrel_loss, FocalLoss
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision import transforms
from json import encoder
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()
from lib.caption_lstm.RelCaptionModel import RelCaptionModel
from tqdm import tqdm

assert conf.vg200_kr_cap

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


def language_eval(preds, test_coco_ids, cache_path):
    import sys

    sys.path.insert(0, "coco_caption")
    # generate target file
    annFile = transform_annos(test_coco_ids)

    from coco_caption.pycocotools.coco import COCO
    from coco_caption.pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    #preds_filt = [p for p in preds if p['image_id'] in valids]
    preds_filt = []
    image_id_filt = []
    for p in preds:
        if p['image_id'] in valids and p['image_id'] not in image_id_filt:
            preds_filt.append(p)
            image_id_filt.append(p['image_id'])
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out


#lang_stats = language_eval(json.load(open('/home/wangwenbin/Method/scene_graph/HRTree-Scene-Graph-Generation-version2/checkpoints/captioning/hrtree-predcls-vg200_kr_caption_print_predictions.json')),
#                           test.coco_ids, '/home/wangwenbin/Method/scene_graph/HRTree-Scene-Graph-Generation-version2/checkpoints/captioning/hrtree-predcls-vg200_kr_caption_cache.json')


### load the caption generator ##################################################
captionGenerator = RelCaptionModel(train.ix_to_word, train.vocab_size, input_encoding_size=300,
                                   rnn_type='lstm', rnn_size=512, num_layers=1, drop_prob_lm=0.5,
                                    seq_length=16, seq_per_img=5, fc_feat_size=4096, att_feat_size=512,
                                   num_relation=conf.num_relation,
                                    object_classes=train.ind_to_classes, predicate_classes=train.ind_to_predicates,
                                   triplet_embed_dim=300, embed_triplet=True)
captionGenerator.cuda()
print(print_para(captionGenerator), flush=True)

if conf.caption_ckpt is not None:
    caption_ckpt = torch.load(conf.caption_ckpt)
    start_epoch = caption_ckpt['epoch']
    if not optimistic_restore(captionGenerator, caption_ckpt['state_dict']):
        start_epoch = -1
else:
    start_epoch = -1

###### now load the relation detector and set it to test mode!!! ###################################
detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order, pick_parent=conf.pick_parent,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
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
                    return_forest=conf.test_forest,
                    need_caption=conf.captioning
                    )

detector.cuda()
ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])
for n, param in detector.named_parameters():
    param.requires_grad = False
detector.eval()
print(print_para(detector), flush=True)
##########################################################################


def get_optim(model, lr):
    params = model.parameters()
    optimizer = optim.Adam(params, weight_decay=0, lr=lr)
    return optimizer

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix]
            else:
                break
        out.append(txt)
    return out


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask =  mask[:, :input.shape[1]]
        input = to_contiguous(input).view(-1, input.shape[2])
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def val_epoch():
    captionGenerator.eval()
    predictions = []

    if conf.test_size > 0:
        max_val_b = conf.test_size // test_loader.batch_size
        test_size = conf.test_size
    else:
        max_val_b = -1
        test_size = len(test)
    dump_img_flag = False
    dir_root = '/'.join(conf.caption_ckpt.split('/')[:-1])
    print(dir_root, test_size)
    if conf.eval_dump:
        image_dir = os.path.join(dir_root+'_images')
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
            dump_img_flag = True
        else:
            if len(glob.glob(os.path.join(image_dir, '*.png'))) != test_size:
                dump_img_flag = True


    loss_sum = 0
    loss_evals = 1e-8
    test_coco_ids = []
    # use test as val
    start = time.time()
    for val_b, batch in enumerate(tqdm(test_loader)):
        if max_val_b > 0 and val_b >= max_val_b:
            break
        result = detector[batch]

        # get the result from scene graph model
        boxes, obj_classes, rels, pred_classes, rel_feats_all, image_fmap, seq_labels, mask_labels, coco_ids = result
        outputs = captionGenerator(rel_feats_all, pred_classes, rels, obj_classes, image_fmap, seq_labels, mask_labels)
        loss = crit(outputs, seq_labels[:, 1:], mask_labels[:, 1:]).data[0]
        loss_sum += loss
        loss_evals += 1

        # get prediction result and evaluate the language criterion
        seq, _ = captionGenerator.sample(rel_feats_all, pred_classes, rels, obj_classes, image_fmap,
                                          beam_size=conf.beam_size, sample_max=conf.sample_max,
                                          temperature=conf.temperature)
        seq = seq.cpu().numpy()  # B x seq_length(16)
        sents = decode_sequence(captionGenerator.vocabs, seq)

        coco_ids = coco_ids.reshape(-1)
        for k, sent in enumerate(sents):
            test_global_id = val_b * test_loader.batch_size + k
            entry = {'image_id': int(coco_ids[k]), 'global_id': int(test_global_id), 'caption': sent}
            test_coco_ids.append(int(coco_ids[k]))
            predictions.append(entry)
            if dump_img_flag: # dump images
                im_size = batch[0][1][k]
                h, w = int(im_size[0]), int(im_size[1])
                image = transforms.ToPILImage()(
                    batch[0][0].cpu().data[k][:, :h, :w] * torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None]
                    + torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]).convert('RGB')
                image.save(os.path.join(image_dir, '%i_%i.png' % (test_global_id, coco_ids[k])))
    end = time.time()
    print("val time/batch = {:.3f}".format(end-start))
    # dump predictions
    if conf.eval_dump:
        json.dump(predictions, open(dir_root+'_print_predictions.json', 'w'))

    lang_stats = language_eval(predictions, test_coco_ids, dir_root+'_cache.json')

    return loss_sum / loss_evals, predictions, lang_stats



print("Evaluation starts now!")

crit = LanguageModelCriterion()

val_loss, predictions, lang_stats = val_epoch()
print("val loss = {:.3f}".format(val_loss))

