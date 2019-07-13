import os
import torch
import torch.utils.data
import csv
import json
import numpy as np
import scipy.ndimage
import pickle
from augmentation_utils import random_transform


def read_csv(csv_file):
    with open(csv_file, 'rb') as fd:
        reader = csv.reader(fd)
        return np.array(list(reader))


class BladderDwiDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, preprocessor=None, augmentation=None, to_tensor=None, caching_data=False,
                 drop_outer_positive=True):
        samples = read_csv(csv_file)
        if drop_outer_positive:
            label = samples[:, 3].astype(np.int)
            samples = samples[label != 2]
        self.num_samples = len(samples)
        self.image = samples[:, 0]
        self.bladder_mask = samples[:, 1]
        self.bladder_bbox = samples[:, 2]
        self.label = samples[:, 3].astype(np.int)
        self.cancer_bboxes = samples[:, 4]
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.to_tensor = to_tensor
        self.caching_data = caching_data
        self._cache = dict()

    def __len__(self):
        return self.num_samples

    def clean_cache(self):
        self._cache = dict()

    def __getitem__(self, idx):

        if self.caching_data and idx in self._cache:
            sample = self._cache[idx]
        else:
            sample = {'image': self.image[idx], 'bladder_mask': self.bladder_mask[idx],
                      'bladder_bbox': self.bladder_bbox[idx],
                      'label': self.label[idx], 'cancer_bboxes': self.cancer_bboxes[idx]}
            if self.preprocessor is not None:
                sample = self.preprocessor(sample)
            if self.caching_data:
                self._cache[idx] = sample
        if self.augmentation is not None:
            sample = self.augmentation(sample)
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)
        return sample

    def get_weights(self):
        w = dict()
        w[0] = 1.0 / np.sum(self.label == 0)  # negative
        w[1] = 1.0 / np.sum(self.label == 1)  # inner positive
        w[2] = 0.0  # outer positive
        weights = []
        for l in self.label:
            weights.append(w[l])
        return weights


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


def lists2slices(tuple_list):
    return [slice(*t) for t in tuple_list]


def read_bladder_bbox(json_file):
    with open(json_file, 'rb') as fd:
        bbox = lists2slices(json.load(fd))
    bbox = bbox[0:2]
    return bbox


def read_cancer_bboxes(filename, image_height, image_width):
    with open(filename, 'r') as f:
        cancer_bboxes = pickle.load(f)

    grid_x, grid_y = np.mgrid[0:image_height, 0:image_width]
    bboxes_image = np.zeros((image_height, image_width))
    for box in cancer_bboxes:
        x = box[0]
        y = box[1]
        r = box[2]
        dist_from_center = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
        mask = dist_from_center < r
        bboxes_image = np.logical_or(bboxes_image, mask)
    return bboxes_image.astype(np.int)


class MyPreprocessor(object):
    def __init__(self, image_root_dir='', cancer_bbox_root_dir='', new_height=None, new_width=None,
                 using_bladder_mask=False, processing_cancer_bboxes=False):
        self.image_root_dir = image_root_dir
        self.cancer_bboxes_root_dir = cancer_bbox_root_dir
        self.new_height = new_height
        self.new_width = new_width
        self.using_bladder_mask = using_bladder_mask
        self.processing_cancer_bboxes = processing_cancer_bboxes

    def _process_one_channel(self, image, bladder_bbox, bladder_mask=None):
        mean = np.mean(image)
        std = max(np.std(image), 1e-9)
        new_image = image[bladder_bbox]
        new_image = (new_image - mean) / std
        if bladder_mask is not None:
            new_image = new_image * bladder_mask
        new_image = resize(new_image, (self.new_height, self.new_width))
        return new_image

    def __call__(self, sample):
        accession_number = sample['image'].split('/', 1)[0]
        image = np.load(os.path.join(self.image_root_dir, sample['image']))
        bladder_bbox = read_bladder_bbox(os.path.join(self.image_root_dir, sample['bladder_bbox']))
        # bladder_bbox = slice(None)
        label = sample['label']
        cancer_bboxes = os.path.join(self.cancer_bboxes_root_dir, sample['cancer_bboxes'])
        if self.using_bladder_mask:
            bladder_mask = np.load(os.path.join(self.image_root_dir, sample['bladder_mask']))
            bladder_mask = bladder_mask[bladder_bbox]
        else:
            bladder_mask = None

        image_0 = self._process_one_channel(image[0], bladder_bbox, bladder_mask)  # ADC
        image_1 = self._process_one_channel(image[1], bladder_bbox, bladder_mask)  # B=0
        image_2 = self._process_one_channel(image[2], bladder_bbox, bladder_mask)  # B=1000

        processed_image = np.stack([image_0, image_1, image_2])

        sample = {'accession_number': accession_number, 'image': processed_image,
                  'label': label, 'cancer_bboxes': cancer_bboxes}
        if self.processing_cancer_bboxes:
            cancer_bboxes_image = read_cancer_bboxes(cancer_bboxes, image.shape[1], image.shape[2])
            cancer_bboxes_image = cancer_bboxes_image[bladder_bbox]
            cancer_bboxes_image = resize(cancer_bboxes_image, (self.new_height, self.new_width))
            sample['cancer_bboxes_image'] = np.expand_dims(cancer_bboxes_image, 0)
        return sample


class MyAugmentation(object):
    def __call__(self, sample):
        # data augmentation
        image = sample['image']
        if 'cancer_bboxes_image' in sample:
            cancer_bboxes_image = sample['cancer_bboxes_image']
            image, cancer_bboxes_image = random_transform(image, cancer_bboxes_image)
            sample['image'] = image
            sample['cancer_bboxes_image'] = cancer_bboxes_image
        else:
            image, _ = random_transform(image)
            sample['image'] = image
        # # flip along x
        # if random.random() < 0.5:
        #     image = np.flip(image, 1)
        # # flip along y
        # if random.random() < 0.5:
        #     image = np.flip(image, 2)
        # # transpose
        # if random.random() < 0.5:
        #     image = image.transpose((0, 2, 1))
        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['image'] = torch.from_numpy(sample['image'].copy())
        sample['label'] = torch.FloatTensor([sample['label']])
        if 'cancer_bboxes_image' in sample:
            sample['cancer_bboxes_image'] = torch.from_numpy(sample['cancer_bboxes_image'].copy())
        return sample


def how_large_bladder():
    # statistic the size of bladder bboxes
    dataset = BladderDwiDataset('/DATA/data/yjgu/bladder/dwi_ax_detection_dataset/all.csv')
    h = []
    w = []
    for sample in dataset:
        bladder_bbox = read_bladder_bbox(
            os.path.join('/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order', sample['bladder_bbox']))
        h.append(bladder_bbox[0].stop - bladder_bbox[0].start)
        w.append(bladder_bbox[1].stop - bladder_bbox[1].start)
        if sample['label'] > 1:
            print(sample)
    print('H', np.min(h), np.max(h), np.mean(h), np.std(h))
    print('W', np.min(w), np.max(w), np.mean(w), np.std(w))


def test():
    image_root_dir = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order'
    cancer_bboxes_root_dir = '/DATA/data/yjgu/bladder/bladder_labels'
    preprocessor = MyPreprocessor(image_root_dir, cancer_bboxes_root_dir, 160, 160, False, True)
    augmentation = MyAugmentation()
    to_tensor = ToTensor()
    dataset = BladderDwiDataset('/DATA/data/yjgu/bladder/dwi_ax_detection_dataset/0_cv_train.csv', preprocessor,
                                augmentation, to_tensor)
    weights = dataset.get_weights()
    print(len(dataset))
    import matplotlib.pyplot as plt

    asample = dataset[286]
    print(asample.keys())
    for k in asample:
        print(k, asample[k])
    print(asample['image'].shape)
    print(np.min(asample['image'].numpy()[1]))
    print(np.mean(asample['image'].numpy()[1]))
    print(np.std(asample['image'].numpy()[1]))
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(asample['image'].numpy()[1], 'gray')
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(asample['image'].numpy()[2], 'gray')
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(asample['image'].numpy()[0], 'gray')
    print(asample['cancer_bboxes_image'].shape)
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(asample['cancer_bboxes_image'].numpy()[0])
    plt.show()


if __name__ == '__main__':
    test()
