import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os
import os.path as osp
import json
import h5py
import queue
from threading import Thread, Lock

# Make sure that caffe is on the python path:
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

EPSILON = 1e-8

data_root = './'
imdb_file = osp.join(data_root, 'imdb_512.h5')
im_h5 = h5py.File(imdb_file, 'r')
im_refs = im_h5['images']
print('imdb: ', im_refs.shape)
im_sizes = np.vstack([im_h5['image_widths'][:], im_h5['image_heights'][:]]).transpose()

# set caffe
caffe.set_mode_gpu()
caffe.set_device(2)
caffe.SGDSolver.display = 0
net = caffe.Net('deploy.prototxt', 'dss_model_released.caffemodel', caffe.TEST)


def add_saliencyMap(im_data, im_sizes, h5_file, net, num_workers=20):
	shape = im_data.shape
	size = shape[-1]
	num_images = shape[0]
	print(num_images)

	image_dset = h5_file.create_dataset('images', (num_images, size, size), dtype=np.float32)

	for i in range(num_images):
		if i % 10000 == 0:
			print('processing %i images...' % i)
		img = im_data[i]
		w, h = im_sizes[i, :]
		img = img[:, :h, :w]  # bgr
		img = np.array(img, dtype=np.float32)
		img = img.transpose((1, 2, 0))  ## chw --> hwc
		img -= np.array((104.00698793,116.66876762,122.67891434))
		img = img.transpose((2, 0, 1))  ## chw

		#lock.acquire()
		# shape for input (data blob is N x C x H x W), set data
		net.blobs['data'].reshape(1, *img.shape)
		net.blobs['data'].data[...] = img
		# run net and take argmax for prediction
		net.forward()
		out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
		out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
		out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
		out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
		out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
		out6 = net.blobs['sigmoid-dsn6'].data[0][0,:,:]
		fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
		res = (out3 + out4 + out5 + fuse) / 4
		res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
		image_dset[i, :h, :w] = res
			#lock.release()
			#q.task_done()

def main():
	# set out saliency db
	out_file = osp.join(data_root, 'saliency_512.h5')
	sal_f = h5py.File(out_file, 'w')
	add_saliencyMap(im_refs, im_sizes, sal_f, net)

if __name__ == '__main__':
	main()


