"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
CO_OCCOUR_PATH = os.path.join(DATA_PATH, 'co_occour_count.npy')

FREQ_WEIGHT = [3, 0.013011092, 0.000447154, 0.000376264, 0.00084523, 0.001177869, 0.004078917, 0.004569696, 0.024838861, 0.004040746, 0.001177869, 0.003740825, 0.001068807, 0.001314197, 0.001843147, 0.000136327, 0.001930397, 0.000534404, 0.000550763, 0.002219411, 0.136970913, 0.025362358, 0.062001723, 0.006958153, 0.001815882, 0.001903131, 0.000927027, 0.000201765, 0.000921574, 0.040260222, 0.106226347, 0.343065295, 0.000970651, 0.002677471, 0.000556216, 0.001592305, 0.00077434, 0.00015814, 0.009422953, 6.54372e-05, 0.012525766, 0.007677962, 0.000834324, 0.009281173, 0.001205135, 0.0006162, 0.003533607, 0.002137614, 0.10962908, 0.012662093, 0.029163168]

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

def vg200_path(fn):
    return os.path.join(DATA_PATH, 'vg200', fn)

def captions_path(fn):
    return os.path.join(DATA_PATH, 'captions', fn)

def vrd_path(fn):
    return os.path.join(DATA_PATH, 'VRD', fn)

CO_OCCOUR_PATH_VG200 = vg200_path('co_occour_count_vg200.npy')

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = stanford_path('Images')
RCNN_CHECKPOINT_FN = stanford_path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

VG200_SGG_FN = vg200_path('VG200-SGG.h5')
VG200_SGG_DICT_FN = vg200_path('VG200-SGG-dicts.json')
SALIENCY_FN = vg200_path('saliency_512.h5')
DEPTH_FN = stanford_path('depth_512.h5')
COCO_PATH = os.path.join(DATA_PATH, 'mscoco')

CAPTIONS_INFO = captions_path('data_vg200kr.json')
CAPTIONS_FN = captions_path('data_vg200kr_label.h5')

## VRD
VRD_TRAIN = vrd_path('HIA/HIA_train.json')
VRD_TEST = vrd_path('HIA/HIA_test.json')
VRD_LABELS = vrd_path('HIA/labels.json')
VRD_TRAIN_IMAGES = vrd_path('sg_dataset/sg_train_images')
VRD_TEST_IMAGES = vrd_path('sg_dataset/sg_test_images')


# =============================================================================
# =============================================================================

# =============================================================================
LOG_SOFTMAX = True
SAMPLE_NUM = 5


# =============================================================================

MODES = ('sgdet', 'sgcls', 'predcls')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding
SAL_SCALE = 512

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_BATCHSIZE = 128

RELPN_BATCHSIZE = 256
RELPN_FG_FRACTION = 0.5

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.coco = None
        self.vg200 = None
        self.vg200_kr = None
        self.vg200_kr_cap = None
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.gt_box = None
        self.mode = None
        self.refine = None
        self.ad3 = False
        self.test = False
        self.adam = False
        self.multi_pred=False
        self.cache = None
        self.model = None
        self.use_proposals=False
        self.use_resnet=False
        self.use_tanh=False
        self.use_bias = False
        self.limit_vision=False
        self.num_epochs=None
        self.old_feats=False
        self.order=None
        self.det_ckpt=None
        self.nl_edge=None
        self.nl_obj=None
        self.hidden_dim=None
        self.pass_in_obj_feats_to_decoder = None
        self.pass_in_obj_feats_to_edge = None
        self.pooling_dim = None
        self.rec_dropout = None
        self.pick_parent = None
        self.isc_thresh = None

        # for relpn
        self.relpn = None
        self.relrank = None
        self.use_CE = None

        # for hierarchy
        self.hir = False

        # visual compare
        self.visual_compare = None

        # two margin super-params
        self.margin1 = None
        self.margin2 = None

        # for tuning the model
        self.rank_input_vis = None
        self.objatt = None
        self.sal_input = None

        # for depth map
        self.use_depth = None

        self.test_forest = None

        self.has_grad = None
        self.use_dist = None

        # captioning task
        self.captioning = None
        self.gcn_captioning = None
        self.num_relation = None
        self.freq_bl = None
        self.caption_ckpt = None

        self.lr_decay_start = None
        self.lr_decay_every = None
        self.lr_decay_rate = None

        self.scheduled_sampling_start = None
        self.scheduled_sampling_increase_every = None
        self.scheduled_sampling_increase_prob = None
        self.scheduled_sampling_max_prob = None

        self.beam_size = None
        self.temperature = None
        self.sample_max = None

        self.grad_clip = None

        self.eval_dump = None
        self.test_size = None
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())


        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

        self.__dict__.update(self.args)

        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if len(self.caption_ckpt) != 0:
            self.caption_ckpt = os.path.join(ROOT_PATH, self.caption_ckpt)
        else:
            self.caption_ckpt = None

        if len(self.cache) != 0:
            self.cache = os.path.join(ROOT_PATH, self.cache)
        else:
            self.cache = None

        if len(self.save_dir) == 0:
            self.save_dir = None
        else:
            self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))

        if self.model not in ('motifnet', 'stanford'):
            raise ValueError("Invalid model {}".format(self.model))


        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        # Options to deprecate
        parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
        parser.add_argument('-vg200', dest='vg200', help='Use VG200 (default to VG)', action='store_true')
        parser.add_argument('-vg200_kr', dest='vg200_kr', help='Use VG200_kr (default to VG)', action='store_true')
        parser.add_argument('-vg200_kr_cap', dest='vg200_kr_cap', help='Use VG200_kr_cap (default to VG)', action='store_true')
        parser.add_argument('-relpn', dest='relpn', help='Need relation proposal (default to False)', action='store_true')
        parser.add_argument('-relrank', dest='relrank', help='Use saliency to sort the relations', action='store_true')
        parser.add_argument('-use_CE', dest='use_CE', help='Use focalCE loss or max margin loss', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')

        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=1)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)

        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=200)
        parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls}', type=str,
                            default='sgdet')
        parser.add_argument('-model', dest='model', help='which model to use? (motifnet, stanford). If you want to use the baseline (NoContext) model, then pass in motifnet here, and nl_obj, nl_edge=0', type=str,
                            default='het')
        parser.add_argument('-old_feats', dest='old_feats', help='Use the original image features for the edges', action='store_true')
        parser.add_argument('-order', dest='order', help='Linearization order for Rois (confidence -default, size, random)',
                            type=str, default='confidence')
        parser.add_argument('-pick_parent', dest='pick_parent', help='how to choose parent (area, isc)', type=str,
                            default='area')
        parser.add_argument('-isc_thresh', dest='isc_thresh', help='the thresh to be a parent', type=float,
                            default=0.9)
        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')
        parser.add_argument('-gt_box', dest='gt_box', help='use gt boxes during training', action='store_true')
        parser.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')
        parser.add_argument('-multipred', dest='multi_pred', help='Allow multiple predicates per pair of box0, box1.', action='store_true')
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=25)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=1)
        parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=2)
        parser.add_argument('-hidden_dim', dest='hidden_dim', help='Num edge layers', type=int, default=256)
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)
        parser.add_argument('-pass_in_obj_feats_to_decoder', dest='pass_in_obj_feats_to_decoder', action='store_true')
        parser.add_argument('-pass_in_obj_feats_to_edge', dest='pass_in_obj_feats_to_edge', action='store_true')
        parser.add_argument('-rec_dropout', dest='rec_dropout', help='recurrent dropout to add', type=float, default=0.0)
        parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
        parser.add_argument('-use_tanh', dest='use_tanh',  action='store_true')
        parser.add_argument('-use_encoded_box', dest='use_encoded_box',  action='store_true')
        parser.add_argument('-use_rl_tree', dest='use_rl_tree',  action='store_true')
        parser.add_argument('-draw_tree', dest='draw_tree',  action='store_true')
        parser.add_argument('-limit_vision', dest='limit_vision',  action='store_true')
        parser.add_argument('-visual_compare', dest='visual_compare', action='store_true')

        parser.add_argument('-hir', dest='hir', action='store_true')

        parser.add_argument('-m1', dest='margin1', type=float, default=0.5)
        parser.add_argument('-m2', dest='margin2', type=float, default=0.4)

        parser.add_argument('-rank_input_vis', dest='rank_input_vis', action='store_true')
        parser.add_argument('-objatt', dest='objatt', action='store_false')
        parser.add_argument('-sal_input', dest='sal_input', type=str, default='both')

        parser.add_argument('-use_depth', dest='use_depth', action='store_true')

        parser.add_argument('-has_grad', dest='has_grad', action='store_true')
        parser.add_argument('-use_dist', dest='use_dist', action='store_true')
        parser.add_argument('-test_forest', dest='test_forest', action='store_true')

        # captioning backend
        parser.add_argument('-captioning', dest='captioning', action='store_true')
        parser.add_argument('-gcn_captioning', dest='gcn_captioning', action='store_true')
        parser.add_argument('-num_relation', dest='num_relation', type=int, default=-1)
        parser.add_argument('-caption_ckpt', dest='caption_ckpt', type=str, default='')

        parser.add_argument('-lr_decay_start', dest='lr_decay_start', type=int, default=0)
        parser.add_argument('-lr_decay_every', dest='lr_decay_every', type=int, default=3)
        parser.add_argument('-lr_decay_rate', dest='lr_decay_rate', type=float, default=0.8)

        parser.add_argument('-scheduled_sampling_start', dest='scheduled_sampling_start', type=int, default=0)
        parser.add_argument('-scheduled_sampling_increase_every', dest='scheduled_sampling_increase_every', type=int, default=5)
        parser.add_argument('-scheduled_sampling_increase_prob', dest='scheduled_sampling_increase_prob', type=float, default=0.05)
        parser.add_argument('-scheduled_sampling_max_prob', dest='scheduled_sampling_max_prob', type=float, default=0.25)

        parser.add_argument('-beam_size', dest='beam_size', type=int, default=1)
        parser.add_argument('-temperature', dest='temperature', type=float, default=1.0)
        parser.add_argument('-sample_max', dest='sample_max', type=int, default=1)

        parser.add_argument('-grad_clip', dest='grad_clip', type=float, default=0.1)
        parser.add_argument('-eval_dump', dest='eval_dump', action='store_true')
        parser.add_argument('-test_size', dest='test_size', help='test size to use (if 0 we wont use val)', type=int,
                            default=-1)
        parser.add_argument('-freq_bl', dest='freq_bl', action='store_true')
        return parser
