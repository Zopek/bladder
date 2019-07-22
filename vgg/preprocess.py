#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os
import csv
import json
import scipy.ndimage
import random

import config as cfg 
import augmentation

class data_preprocess(object):
	def __init__(self):
		self.train_path = cfg.TRAIN_PATH 
		self.test_path = cfg.TEST_PATH 
		self.data_path = cfg.DATA_PATH
		self.label_path = cfg.LABEL_PATH
		self.batch_size = cfg.BATCH_SIZE 
		self.height = cfg.HEIGHT 
		self.width = cfg.WIDTH 
		self.num_class = cfg.NUM_CLASS
		self.train_set = self.load_data('train')
		self.test_set = self.load_data('test')
		self.ind = 0

	def load_data(self, phase = 'train'):
		if phase == 'train':
			with open(self.train_path, 'r') as f:
				reader = list(csv.reader(f))
		elif phase == 'test':
			with open(self.test_path, 'r') as f:
				reader = list(csv.reader(f))

		return reader

	def shuffle(self):
		random.shuffle(self.train_set)

	def size(self, phase = 'train'):
		if phase == 'train':
			size = len(self.train_set)
		elif phase == 'test':
			size = len(self.test_set)

		return size

	def next_batch(self, phase = 'train'):
		if phase == 'train':
			data_set = self.train_set
		elif phase = 'test':
			data_set = self.test_set

		images = []
		labels = []

		for i in range(self.batch_size):

			# dataset[0] = ['D1867766/dwi_ax_0/image_3.npy', 
			#				'D1867766/dwi_ax_0/dilated_mask_3.npy',
			# 				'D1867766/dwi_ax_0/dilated_mask_bbox.json', '0', 
			#				'D1867766/dwi_ax_0/stack0_b0guess/box_label_3.txt']
			image = np.load(os.path.join(self.data_path, data_set[self.ind][0]))
			bladder_bbox = self.read_bladder_bbox(os.path.join(self.data_path, data_set[self.ind][1]))
			cancer_bboxes = dataset[self.ind][3:6]
			sizes = dataset[self.ind][6:9]
			label = int(data_set[self.ind][9])

			if self.ind == self.size(phase) - 1:
				self.ind = 0
			else:
				self.ind += 1

			image_ADC = self.process_one_channel(image[0], bladder_bbox)
			image_b0 = self.process_one_channel(image[1], bladder_bbox)
			image_b1000 = self.process_one_channel(image[2], bladder_bbox)
			image_t2w = self.process_one_channel(image[3], bladder_bbox)
			image_t2wfs = self.process_one_channel(image[4], bladder_bbox)

			cancer_bboxes_image = self.read_cancer_bboxes(cancer_bboxes, sizes)
			cancer_bboxes_image = cancer_bboxes_image[bladder_bbox]
			cancer_bboxes_image = self.resize(cancer_bboxes_image)

			processed_image = np.stack([image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs, cancer_bboxes_image], axis=2)
			# processed_image.shape = [height, width, 3]
			processed_label = np.zeros((self.num_class), np.int32)
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

	def lists2slices(self, tuple_list):

		return [slice(*t) for t in tuple_list]

	def read_bladder_bbox(self, json_file):
		with open(json_file, 'rb') as fd:
			bbox = self.lists2slices(json.load(fd))
		bbox = bbox[0:2]

		return bbox

	def resize(self, image):
		resize_factor = []
		for i, s in enumerate((self.height, self.width)):

			if s is None:
				resize_factor.append(1)
			else:
				resize_factor.append((s + 1e-3) / image.shape[i])
		# resize_factor = (np.round((self.height, self.width)).astype(np.float) + 1e-3) / image.shape
		# +1e-3 to suppress warning of scipy.ndimage.zoom
		new_image = scipy.ndimage.zoom(image, resize_factor, order=1)

		return new_image

	def process_one_channel(self, image, bladder_bbox):
		mean = np.mean(image)
		std = max(np.std(image), 1e-9)
		new_image = image[bladder_bbox]
		new_image = (new_image - mean) / std
		new_image = resize(new_image)

		return new_image

	def read_cancer_bboxes(self, filename, sizes):

	    dwi_size = int(sizes[0])
	    for i in range(len(filename)):

	        if filename[i].split('/')[1] == 'dwi_ax_0':
	            origin_size = int(sizes[0])
	        elif filename[i].split('/')[1] == 't2w_ax_0':
	            origin_size = int(sizes[1])
	        else:
	            assert filename[i].split('/')[1] == 't2wfs_ax_0'
	            origin_size = int(sizes[2])

	        file_path = os.path.join(self.label_path, filename[i])
	        with open(file_path, 'r') as f:
	            cancer_bboxes = pickle.load(f)

	        '''
	        if cancer_bboxes == []:
	            continue
	        '''

	        grid_x, grid_y = np.mgrid[0:origin_size, 0:origin_size]
	        bboxes_image = np.zeros((origin_size, origin_size))
	        for box in cancer_bboxes:
	            
	            x = box[0]
	            y = box[1]
	            r = box[2]
	            dist_from_center = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
	            mask = dist_from_center < r
	            bboxes_image = np.logical_or(bboxes_image, mask)

	        if filename[i].split('/')[1] != 'dwi_ax_0':
	            resize_factor = (np.array((dwi_size, dwi_size)).astype(np.float) + 1e-3) / bboxes_image.shape  # +1e-3 to suppress warning of scipy.ndimage.zoom
	            bboxes_image = scipy.ndimage.interpolation.zoom(bboxes_image.astype(np.float), resize_factor, order=0)

	        if i == 0:
	            bboxes = bboxes_image
	        else:
	            bboxes = np.logical_or(bboxes, bboxes_image)
	        # bboxes.append(bboxes_image)

	    '''
	    if len (bboxes) == 0:
	        bbox = np.zeros((dwi_size, dwi_size))
	    elif len(bboxes) == 1:
	        bbox = bboxes[0]
	    elif len(bboxes) == 2:
	        bbox = np.logical_and(bboxes[0], bboxes[1])
	    elif len(bboxes) == 3:
	        bbox = np.logical_and(np.logical_and(bboxes[0], bboxes[1]), bboxes[2])
	    '''

	    return bboxes.astype(np.int)