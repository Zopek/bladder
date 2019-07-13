#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf 
import tensorlayer as tl
import numpy as np
import os
import csv
import random
import json
from scipy import ndimage
import time
import pickle

import model
import augmentation

data_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
label_path = '/DATA/data/yjgu/bladder/bladder_labels/'

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
	new_image = ndimage.zoom(image, resize_factor, order=1)
	return new_image

def process_one_channel(image, bladder_bbox, height, width):
	mean = np.mean(image)
	std = max(np.std(image), 1e-9)
	new_image = image[bladder_bbox]
	new_image = (new_image - mean) / std
	new_image = resize(new_image, (height, width))
	return new_image

def read_cancer_bbox(root_dir, filename, sizes):

    dwi_size = int(sizes[0])
    for i in range(len(filename)):

        if filename[i].split('/')[1] == 'dwi_ax_0':
            origin_size = int(sizes[0])
        elif filename[i].split('/')[1] == 't2w_ax_0':
            origin_size = int(sizes[1])
        else:
            assert filename[i].split('/')[1] == 't2wfs_ax_0'
            origin_size = int(sizes[2])

        file_path = os.path.join(root_dir, filename[i])
        with open(file_path, 'r') as f:
            cancer_bboxes = pickle.load(f)

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
            bboxes_image = ndimage.interpolation.zoom(bboxes_image.astype(np.float), resize_factor, order=0)

        if i == 0:
            bboxes = bboxes_image
        else:
            bboxes = np.logical_or(bboxes, bboxes_image)
    
    return bboxes.astype(np.int)

def shuffle(dataset):
	random.shuffle(dataset)
	return dataset

def next_batch(dataset, batch_size, height, width, epoch, mode):
	images = []
	labels = []
	accessions = []
	for i in range(batch_size):

		# dataset[0] = ['D1867766/dwi_ax_0/image_3.npy', 
		#				'D1867766/dwi_ax_0/dilated_mask_3.npy',
		# 				'D1867766/dwi_ax_0/dilated_mask_bbox.json', '0', 
		#				'D1867766/dwi_ax_0/stack0_b0guess/box_label_3.txt']
		ind = batch_size * epoch + i
		image = np.load(os.path.join(data_path, dataset[ind][0]))
		bladder_bbox = read_bladder_bbox(os.path.join(data_path, dataset[ind][1]))
		# label = dataset[ind][2]
		cancer_bbox_path = dataset[ind][3:6]
		sizes = dataset[ind][6:]
		accession = dataset[ind][0].split('/')[0] + '_' + dataset[ind][0].split('/')[1].split('.')[0]

		image_ADC = process_one_channel(image[0], bladder_bbox, height, width)
		image_b0 = process_one_channel(image[1], bladder_bbox, height, width)
		image_b1000 = process_one_channel(image[2], bladder_bbox, height, width)
		image_t2w = process_one_channel(image[3], bladder_bbox, height, width)
		image_t2wfs = process_one_channel(image[4], bladder_bbox, height, width)

		processed_image = np.stack([image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs], axis=2)
		# processed_image.shape = [height, width, 3]

		cancer_bbox = read_cancer_bbox(label_path, cancer_bbox_path, sizes)
		cancer_bbox = cancer_bbox[bladder_bbox]
		cancer_bbox = resize(cancer_bbox, (height, width))
		cancer_bbox = np.expand_dims(cancer_bbox, 2)
		# cancer_bbox.shape = [height, width, 1]

		if mode == 'train':
			aug_image, aug_label = augmentation.random_transform(processed_image, cancer_bbox)
		elif mode == 'val':
			aug_image = processed_image
			aug_label = cancer_bbox

		images.append(aug_image)
		labels.append(aug_label)
        accessions.append(accession)

	images = np.asarray(images, dtype=np.float32)
	labels = np.asarray(labels)

	return images, labels, accessions

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], X[:,:,4,np.newaxis],
        y_, y]), size=(1, 7),
        image_path=path)


def main():

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/0_label_1_train_val.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/label_1_test.csv'
    train = []
    val = []
    train_size = 0
    val_size = 0
    batch_size = 8
    height = 160
    width = 160

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

    images, labels, _ = next_batch(val, batch_size, height, width, 0, 'val')
    print(np.sum(np.where(labels.any() != 0 or labels.any() != 1)))
    print(images.shape)
    img = images[0, :, :, 0]
    tl.visualize.save_images(np.stack([images[0, :, :, 0][:,:,np.newaxis], labels[0]], axis=0), [1, 2], '/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/my_unet/unet_brain/label.png')

if __name__ == "__main__":

    main()
