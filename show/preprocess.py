#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import os
import csv
import random
import json
import scipy.ndimage
import pickle

import augmentation

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

senior_path = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order'

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

def next_batch(dataset, batch_size, height, width, epoch):
	images = []
	labels = []

	for i in range(batch_size):

		# dataset[0] = ['D1867766/dwi_ax_0/image_3.npy', 
		#				'D1867766/dwi_ax_0/dilated_mask_3.npy',
		# 				'D1867766/dwi_ax_0/dilated_mask_bbox.json', '0', 
		#				'D1867766/dwi_ax_0/stack0_b0guess/box_label_3.txt']
		ind = batch_size * epoch + i
		image = np.load(os.path.join(senior_path, dataset[ind][0]))
		bladder_bbox = read_bladder_bbox(os.path.join(senior_path, dataset[ind][2]))
		label = dataset[ind][3]
		cancer_bbox_path = os.path.join('/DATA/data/yjgu/bladder/bladder_labels', dataset[ind][4])

		image_ADC = process_one_channel(image[0], bladder_bbox, height, width)
		image_b0 = process_one_channel(image[1], bladder_bbox, height, width)
		image_b1000 = process_one_channel(image[2], bladder_bbox, height, width)

		processed_image = np.stack([image_ADC, image_b0, image_b1000], axis=2)
		# processed_image.shape = [height, width, 3]

		cancer_bbox = read_cancer_bbox(cancer_bbox_path, image.shape[1], image.shape[2], label)
		cancer_bbox = cancer_bbox[bladder_bbox]
		cancer_bbox = resize(cancer_bbox, (height, width))
		cancer_bbox = np.expand_dims(cancer_bbox, 2)
		# cancer_bbox.shape = [height, width, 1]

		aug_image, aug_label = augmentation.random_transform(processed_image, cancer_bbox)

		images.append(aug_image)
		labels.append(aug_label)

	images = np.asarray(images, dtype=np.float32)
	labels = np.asarray(labels)

	return images, labels

def main():
    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 10
    lr = 0.0001 
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 100
    print_freq_step = 100
    height = 160
    width = 160

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_train.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_val.csv'
    train = []
    val = []
    train_size = 0
    val_size = 0
    
    with open(train_path, 'r') as f:
    	reader = csv.reader(f)
    	for i in reader:

    		train.append(i)
    		train_size += 1

    with open(val_path, 'r') as f:
    	reader = csv.reader(f)
    	for i in reader:

    		val.append(i)
    		val_size += 1

    train_epoch = train_size / batch_size
    val_epoch = val_size / batch_size

    ###======================== SHOW DATA ===================================###
    #shuffle(train)
    #shuffle(val)
    images, labels = next_batch(train, batch_size, height, width, 2)
    print(images.shape)
    print(labels.shape)

    image = images[8]
    label = labels[8]


    print(image.shape)
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(image[:,:,0], 'gray')
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(image[:,:,1], 'gray')
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(image[:,:,2], 'gray')
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(label[:,:,0])
    plt.show()
    plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/sample.png')


if __name__ == "__main__":

    main()

