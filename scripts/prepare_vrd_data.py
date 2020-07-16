# ---------------------------------------------------------------
# prepare_vrd_data.py
# Set-up time: 2020/2/27 上午10:53
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import os.path as osp
import mat4py as mp
import cv2
import skimage.io
import numpy as np
import math
import json
from scipy.misc import imread, imresize
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import vrd_path
import h5py


def _clip(a, b):
    return np.array([np.maximum(a[0], b[0]), np.maximum(a[1], b[1]),
                     np.minimum(a[2], b[2]), np.minimum(a[3], b[3])])


def _clip_by_bound(box, h, w):
    return [max(0, min(box[0], w-1)), max(0, min(box[1], h-1)),
            max(0, min(box[2], w-1)), max(0, min(box[3], h-1))]


def findMatched(objects, obj, all_labels, label, max_iou):
    if objects.shape[0] == 0:
        idx = objects.shape[0]
        objects = np.vstack((objects, obj))
        all_labels.append(label)
        return idx, objects, all_labels

    ov = bbox_overlaps(objects, obj)
    if np.max(ov) < max_iou: # this is a new object
        idx = objects.shape[0]
        objects = np.vstack((objects, obj))
        all_labels.append(label)
    else:
        cand_ids = np.where(ov >= max_iou)[0]
        cand_ovs = ov[cand_ids].reshape(-1)
        cand_ids = cand_ids[np.argsort(cand_ovs * -1)]
        mark = False
        for cand_id in cand_ids:
            if all_labels[cand_id] == label:
                idx = cand_id
                mark = True
                break
        if mark is False:
            idx = objects.shape[0]
            objects = np.vstack((objects, obj))
            all_labels.append(label)
    return idx, objects, all_labels



class DataLoader:
    def __init__(self, root, split):
        self._root = root
        self._split = split
        self._image_root = vrd_path('sg_dataset/sg_'+self._split+'_images')
        self._loadLabels()
        self._loadAnnotation()

    def _loadLabels(self):
        mat = mp.loadmat(osp.join(self._root, "mat/predicate.mat"))
        self._relList = mat["predicate"]
        self._relList = ['__background__'] + self._relList
        self._numRelClass = len(self._relList)
        self._relMapping = {}
        for i in range(len(self._relList)):
            self._relMapping[self._relList[i]] = i
        mat = mp.loadmat(osp.join(self._root, "mat/objectListN.mat"))
        self._objList = mat["objectListN"]
        self._objList = ['__background__'] + self._objList
        self._numObjClass = len(self._objList)
        self._objMapping = {}
        for i in range(len(self._objList)):
            self._objMapping[self._objList[i]] = i

    def writeLabels(self):
        with open(osp.join(self._root, 'HIA', 'labels.json'), 'w') as f:
            json.dump({'objects': self._objList, 'predicates': self._relList},  f)

    def _loadAnnotation(self):
        mat = mp.loadmat(osp.join(self._root, "mat/annotation_" + self._split + ".mat"))
        self._annotations = mat["annotation_" + self._split]

    def _getNumImgs(self):
        return len(self._annotations)

    def _getImPath(self, idx):

        return self._annotations[idx]["filename"], osp.join(self._image_root, self._annotations[idx]["filename"])

    def _getNumRel(self):
        numRels = 0
        n = self._getNumImgs()
        for i in range(n):
            rels = self._getRels(i)
            numRels += len(rels)
        return numRels

    def _getRels(self, idx):
        if "relationship" in self._annotations[idx]:
            rels = self._annotations[idx]["relationship"]
            if isinstance(rels, dict):
                rels = [rels]
            return rels
        else:
            return []

    def _outputDB(self, type, data, with_split=True):
        if with_split:
            json.dump(data, open(osp.join(self._root, 'HIA', type + '_' + self._split + ".json"), "w"))
        else:
            json.dump(data, open(osp.join(self._root, 'HIA', type + ".json"), "w"))

    def _bboxTransform(self, bbox, ih, iw):  # [x1, y1, x2, y2]
        return [max(bbox[2], 0), max(bbox[0], 0), min(bbox[3] + 1, iw - 1), min(bbox[1] + 1, ih - 1)]

    def _getRelLabel(self, predicate):
        if not (predicate in self._relMapping):
            return -1
        return self._relMapping[predicate]

    def _getObjLabel(self, predicate):
        if not (predicate in self._objMapping):
            return -1
        return self._objMapping[predicate]

    def _getUnionBBox(self, aBB, bBB, ih, iw, margin=0):
        return [max(0, min(aBB[0], bBB[0]) - margin),
                max(0, min(aBB[1], bBB[1]) - margin),
                min(iw - 1, max(aBB[2], bBB[2]) + margin),
                min(ih - 1, max(aBB[3], bBB[3]) + margin)]

    def _getIntersectionBBox(self, aBB, bBB, im_h, im_w, ratio=1.2):
        # judge whether the two rois intersect with each other or not
        x11, y11, x12, y12 = aBB[0], aBB[1], aBB[2], aBB[3]
        x21, y21, x22, y22 = bBB[0], bBB[1], bBB[2], bBB[3]
        # initialize the intersect box as union box
        union_box = np.array(self._getUnionBBox(aBB, bBB, im_h, im_w))
        intersect_box = union_box.copy()
        iw = np.minimum(x12, x22) - np.maximum(x11, x21) + 1
        ih = np.minimum(y12, y22) - np.maximum(y11, y21) + 1
        # intersect
        if iw > 1 and ih > 1:
            intersect_box = np.array([np.maximum(x11, x21), np.maximum(y11, y21),
                                      np.minimum(x12, x22), np.minimum(y12, y22)])
            w = intersect_box[2] - intersect_box[0] + 1
            h = intersect_box[3] - intersect_box[1] + 1
            nh = h * np.sqrt(ratio)
            nw = w * np.sqrt(ratio)
            deltax = (nw - w) / 2
            deltay = (nh - h) / 2
            intersect_box[0] -= deltax
            intersect_box[1] -= deltay
            intersect_box[2] += deltax
            intersect_box[3] += deltay
            intersect_box = _clip(intersect_box, union_box)
        else:
            w1 = x12 - x11 + 1
            h1 = y12 - y11 + 1
            w2 = x22 - x21 + 1
            h2 = y22 - y21 + 1
            xc1 = (x11 + x12) / 2
            yc1 = (y11 + y12) / 2
            xc2 = (x21 + x22) / 2
            yc2 = (y21 + y22) / 2
            if np.abs(xc1 - xc2) + 1 >= w1 / 2 + w2 / 2 and np.abs(yc1 - yc2) + 1 >= h1 / 2 + h2 / 2:
                intersect_box = np.array([np.minimum(xc1, xc2), np.minimum(yc1, yc2),
                                          np.maximum(xc1, xc2), np.maximum(yc1, yc2)])
            else:
                if np.abs(xc1 - xc2) + 1 < w1 / 2 + w2 / 2:
                    intersect_box = np.array([np.minimum(x11, x21), np.minimum(yc1, yc2),
                                              np.maximum(x12, x22), np.maximum(yc1, yc2)])
                elif np.abs(yc1 - yc2) + 1 < h1 / 2 + h2 / 2:
                    intersect_box = np.array([np.minimum(xc1, xc2), np.minimum(y11, y21),
                                              np.maximum(xc1, xc2), np.maximum(y12, y22)])
        intersect_box = intersect_box.tolist()

        return intersect_box

    def _getRoidb(self, max_iou=0.90):
        n = self._getNumImgs()
        samples = []
        db_idx = 0
        num_images = 4000 if self._split == 'train' else 1000
        max_rel = 0
        num_rel = 0
        num_filtered = 0
        num_keeprel = 0
        max_object = 0
        num_object = 0

        for i in range(n):
            all_object_boxes = np.empty((0, 4), dtype=np.float32)
            all_object_labels = []
            all_rels = []

            rels = self._getRels(i)
            if len(rels) == 0:
                continue
            filename, path = self._getImPath(i)
            im = imread(path)
            if i % 200 == 0:
                print(i, path)
            # handle grayscale
            if im.ndim == 2:
                im = im[:, :, None][:, :, [0, 0, 0]]
            ih = im.shape[0]
            iw = im.shape[1]

            for rel in rels:
                phrase = rel["phrase"]
                rLabel = self._getRelLabel(phrase[1])
                aLabel = self._getObjLabel(phrase[0])
                bLabel = self._getObjLabel(phrase[2])
                aBBox = np.array([c for c in self._bboxTransform(rel["subBox"], ih, iw)]).reshape((1, -1))
                bBBox = np.array([c for c in self._bboxTransform(rel["objBox"], ih, iw)]).reshape((1, -1))
                # identify whether the bbox has been added to the database
                pair_ids = []
                for label, bbox in zip([aLabel, bLabel], [aBBox, bBBox]):
                    idx, all_object_boxes, all_object_labels = findMatched(all_object_boxes, bbox, all_object_labels, label, max_iou)
                    pair_ids.append(idx)
                pair_ids.append(rLabel)
                all_rels.append(pair_ids)

            #filter those with self loop
            all_rels = np.array(all_rels)
            all_rels = all_rels[np.where(all_rels[:, 0] != all_rels[:, 1])[0], :]
            all_rels = all_rels.tolist()

            samples.append({"imPath": filename, "db_idx": db_idx, 'bboxes': all_object_boxes.tolist(), 'gt_classes': all_object_labels,
                            'gt_rels': all_rels})
            max_rel = max(max_rel, len(all_rels))
            num_rel += len(rels)
            num_keeprel += len(all_rels)
            num_filtered += (len(rels) - len(all_rels))
            max_object = max(max_object, len(all_object_boxes))
            num_object += len(all_object_boxes)
            db_idx += 1
        self._outputDB("HIA", samples)
        print('relations: %d, %d, %d, %d' % (max_rel, num_rel, num_keeprel, num_filtered))
        print('objects: %d, %d' % (max_object, num_object))


if __name__ == "__main__":
    vrd_dir = '../data/VRD'
    loader = DataLoader(vrd_dir, "train")
    loader.writeLabels()
    loader._getRoidb()
