import os
import torch
import torch.utils.data
import csv
import json
import numpy as np
from scipy import ndimage
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
            label = samples[:, 2].astype(np.int)
            samples = samples[label != 2]
        self.num_samples = len(samples)
        self.image = samples[:, 0]
        self.bladder_bbox = samples[:, 1]
        self.label = samples[:, 2].astype(np.int)
        self.cancer_bboxes = samples[:, 3:6]
        self.sizes = samples[:, 6:]
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
            sample = {'image': self.image[idx],
                      'bladder_bbox': self.bladder_bbox[idx],
                      'label': self.label[idx], 'cancer_bboxes': self.cancer_bboxes[idx],
                      'size': self.sizes[idx]}
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
    new_image = ndimage.zoom(image, resize_factor, order=1)

    return new_image


def lists2slices(tuple_list):
    return [slice(*t) for t in tuple_list]


def read_bladder_bbox(json_file):
    with open(json_file, 'rb') as fd:
        bbox = lists2slices(json.load(fd))
    bbox = bbox[0:2]
    return bbox


def read_cancer_bboxes(root_dir, filename, sizes):

    dwi_size = int(sizes[0])
    bboxes =[]
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
            bboxes_image = ndimage.interpolation.zoom(bboxes_image.astype(np.float), resize_factor, order=0)

        '''
        if i == 0:
            bboxes = bboxes_image
        else:
            bboxes = np.logical_or(bboxes, bboxes_image)
        '''
        bboxes.append(bboxes_image.astype(np.int))

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

    return bboxes


class MyPreprocessor(object):
    def __init__(self, image_root_dir='', cancer_bbox_root_dir='', new_height=None, new_width=None, 
        processing_cancer_bboxes=True):
        self.image_root_dir = image_root_dir
        self.cancer_bboxes_root_dir = cancer_bbox_root_dir
        self.new_height = new_height
        self.new_width = new_width
        self.processing_cancer_bboxes = processing_cancer_bboxes

    def _process_one_channel(self, image, bladder_bbox):

        mean = np.mean(image)
        std = max(np.std(image), 1e-9)
        new_image = image
        new_image = (new_image - mean) / std
        new_image = resize(new_image, (self.new_height, self.new_width))
        return new_image

    def __call__(self, sample):
        accession_number = sample['image'].split('/', 1)[0]
        image = np.load(os.path.join(self.image_root_dir, sample['image']))
        bladder_bbox = read_bladder_bbox(os.path.join(self.image_root_dir, sample['bladder_bbox']))
        # bladder_bbox = slice(None)
        label = sample['label']
        cancer_bboxes = sample['cancer_bboxes']
        sizes = sample['size']

        image_0 = self._process_one_channel(image[0], bladder_bbox)  # ADC
        image_1 = self._process_one_channel(image[1], bladder_bbox)  # B=0
        image_2 = self._process_one_channel(image[2], bladder_bbox)  # B=1000
        image_3 = self._process_one_channel(image[3], bladder_bbox)  # t2w
        image_4 = self._process_one_channel(image[4], bladder_bbox)  # t2wfs

        processed_image = np.stack([image_0, image_1, image_2, image_3, image_4])


        sample = {'accession_number': accession_number, 'image': processed_image,
                  'label': label}
        if self.processing_cancer_bboxes:
            cancer_bboxes_image = read_cancer_bboxes(self.cancer_bboxes_root_dir, cancer_bboxes, sizes)
            cancer_bboxes_image = cancer_bboxes_image
            cancer_bboxes_image = np.array([resize(cancer_bboxes_image[0], (self.new_height, self.new_width)),
            resize(cancer_bboxes_image[1], (self.new_height, self.new_width)),
            resize(cancer_bboxes_image[2], (self.new_height, self.new_width))])
            sample['cancer_bboxes_image'] = cancer_bboxes_image

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
    image_root_dir = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
    cancer_bboxes_root_dir = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels'
    preprocessor = MyPreprocessor(image_root_dir, cancer_bboxes_root_dir, 160, 160, True)
    augmentation = MyAugmentation()
    to_tensor = ToTensor()
    dataset = BladderDwiDataset('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/0_cv_train.csv', preprocessor,
                                None, to_tensor, False, False)
    weights = dataset.get_weights()
    print(len(dataset))
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    for i in range(1000):

        asample = dataset[i]
        if asample['label'] == 1:
            fig = plt.figure()
            ax = fig.add_subplot(4, 2, 1)
            ax.imshow(asample['image'].numpy()[0], 'gray')
            ax = fig.add_subplot(4, 2, 2)
            ax.imshow(asample['image'].numpy()[1], 'gray')
            ax = fig.add_subplot(4, 2, 3)
            ax.imshow(asample['image'].numpy()[2], 'gray')
            ax = fig.add_subplot(4, 2, 4)
            ax.imshow(asample['image'].numpy()[3], 'gray')
            ax = fig.add_subplot(4, 2, 5)
            ax.imshow(asample['image'].numpy()[4], 'gray')
            # print(asample['cancer_bboxes_image'].shape)
            ax = fig.add_subplot(4, 2, 6)
            ax.imshow(asample['cancer_bboxes_image'][0].numpy())
            ax = fig.add_subplot(4, 2, 7)
            ax.imshow(asample['cancer_bboxes_image'][1].numpy())
            ax = fig.add_subplot(4, 2, 8)
            ax.imshow(asample['cancer_bboxes_image'][2].numpy())
            plt.show()
            plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/sample/my_unet/origin_sample_{}_{}.png'.format(i, asample['accession_number']))


if __name__ == '__main__':
    test()
