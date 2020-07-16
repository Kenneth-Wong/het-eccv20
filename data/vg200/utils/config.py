import os
import os.path as osp
import json
import h5py
import numpy as np

ROOT_DIR = '/home/wangwenbin/Method/scene_graph/het-eccv20'
DATA_DIR = osp.join(ROOT_DIR, 'data')
CACHE_DIR = osp.join(DATA_DIR, 'cache')
VG_DIR = osp.join(DATA_DIR, 'stanford_filtered')
IMAGE_ROOT_DIR = osp.join(VG_DIR, 'Images')
IMAGE_DIR1 = osp.join(IMAGE_ROOT_DIR, 'VG_100K')
IMAGE_DIR2 = osp.join(IMAGE_ROOT_DIR, 'VG_100K_2')
imdb_file = osp.join(VG_DIR, 'imdb_1024.h5')
imdb512_file = osp.join(VG_DIR, 'imdb_512.h5')
roidb_file = osp.join(VG_DIR, 'VG-SGG.h5')
dict_file = osp.join(VG_DIR, 'VG-SGG-dicts.json')
meta_file = osp.join(VG_DIR, 'image_data.json')
sal_file = osp.join(VG_DIR, 'saliency_512.h5')
#triplet_match_file = osp.join(DATA_DIR, 'triplet_match.json')
#merge_rel_file = osp.join(RAW_DIR, 'merge_relationships.json')

GLOVE_DIR = osp.join(DATA_DIR, 'GloVe')

NEW_VG_DIR = osp.join(DATA_DIR, 'vg200')
cap_to_sg_file = osp.join(NEW_VG_DIR, 'captions_to_sg.json')

objects_file = osp.join(NEW_VG_DIR, 'objects.json')
rels_file = osp.join(NEW_VG_DIR, 'relationships.json')
obj_alias_file = osp.join(NEW_VG_DIR, 'object_alias.txt')
pred_alias_file = osp.join(NEW_VG_DIR, 'relationship_alias.txt')
cleanse_objects_file = osp.join(NEW_VG_DIR, 'cleanse_objects.json')
cleanse_rels_file = osp.join(NEW_VG_DIR, 'cleanse_relationships.json')
cleanse_triplet_match_file = osp.join(NEW_VG_DIR, 'cleanse_triplet_match.json')

object_list_file = osp.join(NEW_VG_DIR, 'object_list.txt')
predicate_list_file = osp.join(NEW_VG_DIR, 'predicate_list.txt')
predicate_stem_file = osp.join(NEW_VG_DIR, 'predicate_stem.txt')

# ====================== output VG 200-80 roidb

NEW_VG_SGG_FILE = osp.join(NEW_VG_DIR, 'VG200-SGG.h5')
NEW_VG_SGG_DICT = osp.join(NEW_VG_DIR, 'VG200-SGG-dicts.json')











