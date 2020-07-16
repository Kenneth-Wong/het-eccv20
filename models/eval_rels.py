from dataloaders.visual_genome import VGDataLoader, VG
from dataloaders.visual_genome200 import VG200
from dataloaders.visual_genome200 import VGDataLoader as VGDataLoader200
from dataloaders.visual_genome200_keyrel import VG200_Keyrel
from dataloaders.visual_genome200_keyrel import VGDataLoader as VGDataLoader200_KR
import numpy as np
import torch

from config import ModelConfig, ROOT_PATH
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
import pickle

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

train, val, test = VGdata.splits(num_im=-1, num_val_im=conf.val_size, filter_duplicate_rels=True,
                                 use_proposals=conf.use_proposals,
                                 filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    #pass
    val = test
train_loader, val_loader = VGLoader.splits(train, val, mode='rel',
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
                    dbname=dbname,
                    sal_input=conf.sal_input,
                    use_depth=conf.use_depth,
                    return_forest=conf.test_forest
                    )

detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])
# if conf.mode == 'sgdet':
#     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

all_pred_entries = []

def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):
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
        all_pred_entries.append(pred_entry)

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred,
                                               num_predicates=80 if conf.vg200 or conf.vg200_kr else 50)


if conf.cache is not None and os.path.exists(conf.cache):
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache, 'rb') as f:
        all_pred_entries = pkl.load(f)
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
            'gt_key_rels': val.key_rel_idxes[i].copy() if conf.vg200_kr else None
        }
        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    evaluator[conf.mode].print_stats()
else:
    detector.eval()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
        #if val_b and val_b % 5000 == 0:
    evaluator[conf.mode].print_stats()

    if conf.cache is not None:
        with open(conf.cache, 'wb') as f:
            pkl.dump(all_pred_entries, f)

    if len(evaluator[conf.mode].tree_depths) > 0:
        with open('/'.join(conf.ckpt.split('/')[:-1])+'_tree_depths.txt', 'w') as f:
            for depth in evaluator[conf.mode].tree_depths:
                f.write(str(depth)+'\n')
            #pkl.dump(evaluator[conf.mode].tree_depths, f)

        with open('/'.join(conf.ckpt.split('/')[:-1]) + '_predrel_depth_scores.pkl', 'wb') as f:
            pickle.dump(evaluator[conf.mode].predrel_treedeep_scores_dict[conf.mode], f)

            #for pair_depth, pair_scores in evaluator[conf.mode].predrel_treedeep_scores_dict[conf.mode].items():
            #    f.write('{} {}:{}\n'.format(pair_depth[0], pair_depth[1], ' '.join(list(map(str, pair_scores)))))
