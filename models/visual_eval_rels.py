from dataloaders.visual_genome import VGDataLoader, VG
from dataloaders.visual_genome200 import VG200
from dataloaders.visual_genome200 import VGDataLoader as VGDataLoader200
from dataloaders.visual_genome200_keyrel import VG200_Keyrel
from dataloaders.visual_genome200_keyrel import VGDataLoader as VGDataLoader200_KR
from dataloaders.visual_genome200_keyrel_captions import VG200_Keyrel_captions
from dataloaders.visual_genome200_keyrel_captions import VGDataLoader as VGDataLoader200_KR_cap
import numpy as np
import torch
from PIL import Image
from config import ModelConfig, ROOT_PATH
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
from torchvision import transforms

conf = ModelConfig()
if conf.model == 'motifnet':
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
if conf.test:
    #pass
    val = test
train_loader, val_loader = VGLoader.splits(train, val, mode='rel',
                                           batch_size=conf.batch_size,
                                           num_workers=conf.num_workers,
                                           num_gpus=conf.num_gpus)

#'''
detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
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
                    dbname=dbname
                    )

detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])
#'''
# if conf.mode == 'sgdet':
#     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

all_pred_entries = []
all_gt_entries = []

def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, rel_rank_scores_i, gt_boxes, gt_classes, gt_rels) in enumerate(det_res):
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
            'rel_rank_scores': rel_rank_scores_i
        }
        all_pred_entries.append(pred_entry)

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


gt_cache = os.path.join(ROOT_PATH, dbname+'_gt4VC_allTestIM.pkl')
if not os.path.isdir(os.path.join(ROOT_PATH, 'vg200krTestImgs')):
    os.makedirs(os.path.join(ROOT_PATH, 'vg200krTestImgs'))
if not os.path.exists(gt_cache):
    print('Writing gt entries...')
    with open(gt_cache, 'wb') as f:
        for val_b, batch in enumerate(tqdm(val_loader)):
            #if val_b > 3:
            #    break
            im_size = batch[0][1][0]
            h, w = int(im_size[0]), int(im_size[1])
            sents = None
            if conf.vg200_kr_cap:
                caption_labels = batch[0][11].data.numpy()[:, 1:]
                sents = []
                for seq in caption_labels:
                    sent = []
                    for lab in seq:
                        if lab > 0:
                            sent.append(test.ix_to_word[str(lab)])
                    sents.append(sent)
            image = transforms.ToPILImage()(batch[0][0].data[0][:, :h, :w] * torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None]
                                        + torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]).convert('RGB')
            image.save(os.path.join(ROOT_PATH, 'vg200krTestImgs', '%i.png'%val_b))
            gt_entry = {
                'images': image,
                'sal_map': Image.fromarray(np.uint8(np.asarray(transforms.Resize(592)(transforms.ToPILImage()(batch[0][8].data[0])))[:h, :w])),
                'depth_map': transforms.ToPILImage()(batch[0][10].data[0][:, :h, :w]),
                'captions': sents,
                'gt_classes': val.gt_classes[val_b].copy(),
                'gt_relations': val.relationships[val_b].copy(),
                'gt_boxes': val.gt_boxes[val_b].copy(),
                'gt_key_rels': val.key_rel_idxes[val_b].copy() if conf.vg200_kr or conf.vg200_kr_cap else None
                }
            all_gt_entries.append(gt_entry)

        pkl.dump(all_gt_entries, f)

if conf.cache is not None and not os.path.exists(conf.cache):
    print("Writing {}! Loading from it".format(conf.cache))
    with open(conf.cache, 'wb') as f:
        detector.eval()
        for val_b, batch in enumerate(tqdm(val_loader)):
            #if val_b < 500:
            det_res = detector[batch]
            if conf.num_gpus == 1:
                det_res = [det_res]
            for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, rel_rank_scores_i, gt_boxes, gt_classes,
                gt_rels, _) in enumerate(det_res):
                pred_entry = {
                    'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
                    'pred_classes': objs_i,
                    'pred_rel_inds': rels_i,
                    'obj_scores': obj_scores_i,
                    'rel_scores': pred_scores_i,  # hack for now.
                    'rel_rank_scores': rel_rank_scores_i
                }
                all_pred_entries.append(pred_entry)
        pkl.dump(all_pred_entries, f)


