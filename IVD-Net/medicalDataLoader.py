from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

from scipy import ndimage
import numpy as np
import os
import json
import pickle

import augmentation

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

class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, dataset, data_path, label_path, height, width, transform=None, mask_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.dataset = dataset
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.mask_transform = mask_transform
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # dataset[0] = ['D1867766/dwi_ax_0/image_3.npy', 
        #               'D1867766/dwi_ax_0/dilated_mask_3.npy',
        #               'D1867766/dwi_ax_0/dilated_mask_bbox.json', '0', 
        #               'D1867766/dwi_ax_0/stack0_b0guess/box_label_3.txt']
        image = np.load(os.path.join(self.data_path, self.dataset[index][0]))
        bladder_bbox = read_bladder_bbox(os.path.join(self.data_path, self.dataset[index][1]))
        # label = dataset[ind][2]
        cancer_bbox_path = self.dataset[index][3:6]
        sizes = self.dataset[index][6:]

        image_ADC = process_one_channel(image[0], bladder_bbox, self.height, self.width)
        image_b0 = process_one_channel(image[1], bladder_bbox, self.height, self.width)
        image_b1000 = process_one_channel(image[2], bladder_bbox, self.height, self.width)
        image_t2w = process_one_channel(image[3], bladder_bbox, self.height, self.width)
        image_t2wfs = process_one_channel(image[4], bladder_bbox, self.height, self.width)

        processed_image = np.stack([image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs], axis=2)
        # processed_image.shape = [height, width, 3]

        cancer_bbox = read_cancer_bbox(self.label_path, cancer_bbox_path, sizes)
        cancer_bbox = cancer_bbox[bladder_bbox]
        cancer_bbox = resize(cancer_bbox, (self.height, self.width))
        cancer_bbox = np.expand_dims(cancer_bbox, 2)
        # cancer_bbox.shape = [height, width, 1]

        if self.mode == 'train':
            aug_image, aug_label = augmentation.random_transform(processed_image, cancer_bbox)
        elif self.mode == 'val':
            aug_image = processed_image
            aug_label = cancer_bbox

        images = np.asarray(aug_image, dtype=np.float32)
        labels = np.asarray(aug_label)

        image_ADC = images[:,:,0]
        image_b0 = images[:,:,1]
        image_b1000 = images[:,:,2]
        image_t2w = images[:,:,3]
        image_t2wfs = images[:,:,4]
        label = labels[:,:,0]

        if self.transform:
            image_ADC = self.transform(image_ADC)
            image_b0 = self.transform(image_b0)
            image_b1000 = self.transform(image_b1000)
            image_t2w = self.transform(image_t2w)
            image_t2wfs = self.transform(image_t2wfs)
            label = self.mask_transform(label)

        return [image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs, label]