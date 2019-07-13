# -*- coding: UTF-8 -*-

import tensorflow as tf 
import numpy as np
import os
import csv
import random
import json
import scipy.ndimage
import pickle

import augmentation

data_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'

def lists2slices(tuple_list):
	return [slice(*t) for t in tuple_list]

def read_bladder_bbox(json_file):
	with open(json_file, 'rb') as fd:
		bbox = lists2slices(json.load(fd))
	bbox = bbox[0:2]
	return bbox

def resize(image, new_shape):
	resize_factor = []
	for i, s in enumerate(new_shape):

		if s is None:
			resize_factor.append(1)
		else:
			resize_factor.append((s + 1e-3) / image.shape[i])
	# resize_factor = (np.round(new_shape).astype(np.float) + 1e-3) / image.shape
	# +1e-3 to suppress warning of scipy.ndimage.zoom
	new_image = scipy.ndimage.zoom(image, resize_factor, order=1)
	return new_image

def process_one_channel(image, bladder_bbox, height, width):
	mean = np.mean(image)
	std = max(np.std(image), 1e-9)
	new_image = image[bladder_bbox]
	new_image = (new_image - mean) / std
	new_image = resize(new_image, (height, width))
	return new_image

def read_cancer_bbox(filename, image_height, image_width, label):
	with open(filename, 'r') as f:
		cancer_bboxes = pickle.load(f)

	bboxes_image = np.zeros((image_height, image_width))
	grid_x, grid_y = np.mgrid[0:image_height, 0:image_width]
	for box in cancer_bboxes:

		x = box[0]
		y = box[1]
		r = box[2]
		dist_from_center = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
		mask = dist_from_center < r
		bboxes_image = np.logical_or(bboxes_image, mask)

	return bboxes_image.astype(np.int)

def shuffle(dataset):
	random.shuffle(dataset)
	return dataset

def next_batch(dataset, batch_size, height, width, epoch, class_num, phase):
	images = []
	labels = []

	for i in range(batch_size):

		# dataset[0] = ['D1867766/dwi_ax_0/image_3.npy', 
		#				'D1867766/dwi_ax_0/dilated_mask_3.npy',
		# 				'D1867766/dwi_ax_0/dilated_mask_bbox.json', '0', 
		#				'D1867766/dwi_ax_0/stack0_b0guess/box_label_3.txt']
		ind = batch_size * epoch + i
		image = np.load(os.path.join(data_path, dataset[ind][0]))
		bladder_bbox = read_bladder_bbox(os.path.join(data_path, dataset[ind][1]))
		label = int(dataset[ind][2])

		image_ADC = process_one_channel(image[0], bladder_bbox, height, width)
		image_b0 = process_one_channel(image[1], bladder_bbox, height, width)
		image_b1000 = process_one_channel(image[2], bladder_bbox, height, width)
		image_t2w = process_one_channel(image[3], bladder_bbox, height, width)
		image_t2wfs = process_one_channel(image[4], bladder_bbox, height, width)

		processed_image = np.stack([image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs], axis=2)
		# processed_image.shape = [height, width, 3]
		processed_label = np.zeros((class_num), np.int32)
		processed_label[label] = 1
		# processed_label[int(label)] = 1

		if phase == 'train':
			aug_image, _ = augmentation.random_transform(processed_image)
		elif phase == 'test':
			aug_image = processed_image

		images.append(aug_image)
		labels.append(processed_label)

	images = np.asarray(images, dtype=np.float32)
	labels = np.asarray(labels)

	return images, labels






