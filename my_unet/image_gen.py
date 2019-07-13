from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import pickle
import os
import csv
import json
import scipy.ndimage
from image_util import BaseDataProvider

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

def GetDataset(path, batch_size):
    dataset = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            dataset.append(i)
    
    iters = int(len(dataset) / batch_size)

    return dataset, iters

class BladderDataProvider(BaseDataProvider):
    channels = 3
    n_class = 2
    
    def __init__(self, nx, ny, dataset):
        super(BladderDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.dataset = dataset
        self.ind = -1

        np.random.shuffle(self.dataset)

    def _process_one_channel(self, image, bladder_bbox):
        mean = np.mean(image)
        std = max(np.std(image), 1e-9)
        new_image = image[bladder_bbox]
        new_image = (new_image - mean) / std
        new_image = resize(new_image, (self.nx, self.ny))
        
        return new_image

    def _cylce_file(self):
        self.ind += 1
        if self.ind >= len(self.dataset):
            self.ind = 0 
            np.random.shuffle(self.dataset)

    def _next_data(self):
        self._cylce_file()

        image = np.load(os.path.join(senior_path, self.dataset[self.ind][0]))
        bladder_bbox = read_bladder_bbox(os.path.join(senior_path, self.dataset[self.ind][2]))
        label = self.dataset[self.ind][3]
        cancer_bbox_path = os.path.join('/DATA/data/yjgu/bladder/bladder_labels', self.dataset[self.ind][4])

        image_ADC = self._process_one_channel(image[0], bladder_bbox)
        image_b0 = self._process_one_channel(image[1], bladder_bbox)
        image_b1000 = self._process_one_channel(image[2], bladder_bbox)

        processed_image = np.stack([image_ADC, image_b0, image_b1000], axis=2)
        # processed_image.shape = [self.nx, self.ny, 3]

        cancer_bbox = read_cancer_bbox(cancer_bbox_path, image.shape[1], image.shape[2], label)
        cancer_bbox = cancer_bbox[bladder_bbox]
        cancer_bbox = resize(cancer_bbox, (self.nx, self.ny))
        # cancer_bbox = np.expand_dims(cancer_bbox, 2)

        return processed_image, cancer_bbox

def main():
    height = 160
    width = 160
    batch_size = 10
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_train.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_val.csv'

    dataset, iters = GetDataset(train_path, batch_size)
    generator = BladderDataProvider(height, width, dataset)

    image, label =generator(1)

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    print(image.shape)
    print(label.shape)
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(image[0][:,:,0], 'gray')
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(image[0][:,:,1], 'gray')
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(image[0][:,:,2], 'gray')
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(label[0][:,:,0], 'gray')
    plt.show()
    plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/sample2.png')

if __name__ == "__main__":

    main()