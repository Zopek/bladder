#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import os
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from scipy import ndimage
import pydicom as dicom
import re
import collections
import csv
import tensorlayer as tl

import preprocess_util

B_LOW = 0
B_HIGH = 1000
NEW_SPACING_SIZE = 1.25
NEW_SPACING = np.array([NEW_SPACING_SIZE, NEW_SPACING_SIZE, NEW_SPACING_SIZE])  # unit is mm
MIN_BLADDER_VOLUMN = 25000  # unit is mm^3

ADC_THRESHOLD_SCALE = 0.7
DWI_LOW_THRESHOLD_SCALE = 0.4

DILATED_PIXELS = int(np.round(10 / NEW_SPACING_SIZE))

def resample(image, spacing, new_spacing):

    # print 'start ', image.shape
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor) + 1e-3
    real_resize_factor = new_shape / image.shape
    real_new_spacing = spacing / real_resize_factor
    new_image = ndimage.interpolation.zoom(image, real_resize_factor, order=2)
    # print 'end ', new_image.shape
    return new_image, real_new_spacing

def _sort_by_slice_location(slice_datasets):

    slice_locations = [d.SliceLocation for d in slice_datasets]
    return [d for (s, d) in sorted(zip(slice_locations, slice_datasets))]

def _merge_slice_pixel_arrays(slice_datasets):

    first_dataset = slice_datasets[0]
    num_rows = first_dataset.Rows
    num_columns = first_dataset.Columns
    num_slices = len(slice_datasets)
    sorted_slice_datasets = _sort_by_slice_location(slice_datasets)
    dtype = first_dataset.pixel_array.dtype
    voxels = np.empty((num_columns, num_rows, num_slices), dtype=dtype)
    for k, dataset in enumerate(sorted_slice_datasets):
        voxels[:, :, k] = dataset.pixel_array.T

    return voxels

def read_dcm(mode_path):

    images_list = []
    images_spacing = None
    for stack in os.listdir(mode_path):

        stack_path = os.path.join(mode_path, stack)
        dcm_list = []
        space = []
        for file in os.listdir(stack_path):

            if file.endswith(".dcm"):
                file_path = os.path.join(stack_path, file)
                dcm = dicom.read_file(file_path)
                dcm_list.append(dcm)

            break
        assert len(dcm_list) >= 1
        images = _merge_slice_pixel_arrays(dcm_list)

        space.append(dcm_list[0].PixelSpacing[0])
        space.append(dcm_list[0].PixelSpacing[1])
        spacing = np.array(space + [dcm_list[0].SpacingBetweenSlices])

        assert spacing is not None
        if images_spacing is None:
            images_spacing = spacing
        else:
            assert np.array_equal(images_spacing, spacing)

        images[images < 1] = 1
        images_list.append(images)

    return images_list, images_spacing

def read_stack(stack_path):
    ds_list = []
    for file_name in os.listdir(stack_path):
        if file_name.endswith('.dcm'):
            ds = dicom.read_file(os.path.join(stack_path, file_name))
            ds_list.append(ds)
    assert len(ds_list) >= 1
    voxel_array = _merge_slice_pixel_arrays(ds_list)
    # add by me
    space = []
    space.append(ds_list[0].PixelSpacing[0])
    space.append(ds_list[0].PixelSpacing[1])
    # modified
    spacing = np.array(space + [ds_list[0].SpacingBetweenSlices])
    return voxel_array, spacing

def read_dwi(dwi_series_path):

    b_value_list = []
    dwi_list = []
    dwi_spacing = None

    for stack_dir in os.listdir(dwi_series_path):

        [_, b_value_str] = stack_dir.split('_')
        if 'None' not in b_value_str:
            b_value = re.findall(r'\d+', b_value_str)
            assert len(b_value) == 1
            b_value = int(b_value[0])
            stack_path = os.path.join(dwi_series_path, stack_dir)
            dwi_array, spacing = read_stack(stack_path)
            assert spacing is not None
            if dwi_spacing is None:
                dwi_spacing = spacing
            else:
                assert np.array_equal(dwi_spacing, spacing)
            dwi_array[dwi_array < 1] = 1
            b_value_list.append(b_value)
            dwi_list.append(dwi_array)
    dwi_ordered_dict = collections.OrderedDict(sorted(zip(b_value_list, dwi_list)))

    return dwi_ordered_dict, dwi_spacing

def preprocess_dwi(mode_path):

    dwi_ordered_dict, spacing = read_dwi(mode_path)
    b_values = np.array(dwi_ordered_dict.keys())
    nearest_b_low = preprocess_util.find_nearest(b_values, B_LOW)
    nearest_b_high = preprocess_util.find_nearest(b_values, B_HIGH)
    assert np.abs(nearest_b_low - B_LOW) <= 500
    assert np.abs(nearest_b_high - B_HIGH) <= 500
    tmp_dwi_dict = {nearest_b_low: dwi_ordered_dict[nearest_b_low], nearest_b_high: dwi_ordered_dict[nearest_b_high]}
    image = preprocess_util.dwi2adc(tmp_dwi_dict)
    if nearest_b_low == B_LOW:
        dwi_low = dwi_ordered_dict[B_LOW]
    else:
        dwi_low = preprocess_util.calculate_dwi(image, dwi_ordered_dict[nearest_b_low], nearest_b_low, B_LOW)

    if nearest_b_high == B_HIGH:
        dwi_high = dwi_ordered_dict[B_HIGH]
    else:
        dwi_high = preprocess_util.calculate_dwi(image, dwi_ordered_dict[nearest_b_high], nearest_b_high, B_HIGH)
    assert dwi_low.shape == image.shape
    assert dwi_high.shape == image.shape

    image[np.isinf(image) | np.isnan(image) | (image < 0)] = 0
    resampled_adc, new_spacing = preprocess_util.resample(image, spacing, NEW_SPACING)
    resampled_dwi_low = preprocess_util.resize(dwi_low, resampled_adc.shape)

    voxel_volumn = 1
    for sp in new_spacing:
        voxel_volumn *= sp

    def get_bladder_mask(img, scale):

        denoised = preprocess_util.denoise(img.astype(np.float))
        threshold = preprocess_util.get_bladder_threshold(denoised, voxel_volumn, MIN_BLADDER_VOLUMN, scale)

        return denoised > threshold

    mask_adc = get_bladder_mask(resampled_adc, ADC_THRESHOLD_SCALE)
    mask_dwi_low = get_bladder_mask(resampled_dwi_low, DWI_LOW_THRESHOLD_SCALE)
    mask = np.logical_and(mask_adc, mask_dwi_low)
    mask = preprocess_util.postprocess_bladder_mask(mask)

    dilated_mask = preprocess_util.get_dilated_mask(mask, DILATED_PIXELS)
    dilated_mask = preprocess_util.resize(dilated_mask.astype(np.float), image.shape)
    dilated_mask = dilated_mask > 0.5

    return dilated_mask, image

def preprocess_others(mode_path):

    image, spacing = read_dcm(mode_path)
    image = image[0]
    # only dwi and dce have more than one stacks

    image[np.isinf(image) | np.isnan(image) | (image < 0)] = 0

    return image

def read_cancer_bbox(filename, image_height, image_width):
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


def main():


    data_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
    record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/all_sizes.csv'

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    '''
    labeled_list = []
    images = []
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)

        for item in reader:

            if item[9] in labeled_list:
                continue
            labeled_list.append(item[9])
            accession = item[0].split('/')[0]

            series_path = os.path.join(data_path, accession, 'dwi_ax_0')
            stack = os.listdir(series_path)[0]
            stack_path = os.path.join(series_path, stack)
            image_path = os.path.join(stack_path, 'I_{}.dcm'.format(item[0].split('_')[1].split('.')[0]))
            ds = dicom.read_file(image_path)
            images.append(ds.pixel_array.T)

    fig = plt.figure()
    for i in range(len(images)):

        ax = fig.add_subplot(4,2,i+1)
        ax.imshow(images[i], 'gray')
        print(labeled_list[i])
    plt.show()
    plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/sample_paper/periods_dwi_images.png')
    '''
    
    image_path = os.path.join(data_path, 'D0564603', 'image_15.npy')
    image = np.load(image_path)
    fig = plt.figure()
    for i in range(image.shape[0]):
        
        ax = fig.add_subplot(1,5,i+1)
        ax.imshow(image[i], 'gray')
    plt.show()
    plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/sample_paper/D0564603_processed_images.png')
    
    '''
    accession = 'D0564603'
    image_path = []
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/dce_ax_0/stack0/I_15.dcm')
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/dce_ax_0/stack1/I_15.dcm')
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/dce_ax_0/stack2/I_15.dcm')
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/dce_ax_0/stack3/I_15.dcm')
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/dwi_ax_0/stack0_b0guess/I_15.dcm')
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/dwi_ax_0/stack1_b1000guess/I_15.dcm')
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/t2w_ax_0/stack0/I_15.dcm')
    image_path.append('/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D0564603/t2wfs_ax_0/stack0/I_15.dcm')

    fig = plt.figure()
    for i in range(1,9):

        ds = dicom.read_file(image_path[i-1])
        image = ds.pixel_array.T
        ax = fig.add_subplot(2,4,i)
        ax.imshow(image, 'gray')

    plt.show()
    plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/sample_paper/D0564603_images.png')
    '''
    '''
    accession_list = []
    i = 0

    with open(record_path, 'rb') as f:
        reader = csv.reader(f)

        for item in reader:

            if item[2] == '1':
                accession = item[0].split('/')[0]
                if accession in accession_list:
                    continue
                accession_list.append(accession)

                image_path = os.path.join(data_path, accession, 'dwi_ax_0', item[0].split('/')[1])
                image = np.load(image_path)

                fig = plt.figure()
                ax = fig.add_subplot(1,3,1)
                ax.imshow(image[0], 'gray')
                ax = fig.add_subplot(1,3,2)
                ax.imshow(image[1], 'gray')
                ax = fig.add_subplot(1,3,3)
                ax.imshow(image[2], 'gray')
                plt.show()
                plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/sample_paper/b0_b1000_{}_{}.png'.format(i, accession))

                i += 1
                if i == 10:
                    break
    '''
    '''
##################################################save label########################################
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/0_label_1_train_val.csv'
    label_path = '/DATA/data/yjgu/bladder/bladder_labels/'

    train = []
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            train.append(i)

    cancer_bbox_path = train[0][3:6]
    sizes = train[0][6:]

    dwi_size = int(sizes[0])
    for i in range(len(cancer_bbox_path)):

        if cancer_bbox_path[i].split('/')[1] == 'dwi_ax_0':
            origin_size = int(sizes[0])
        elif cancer_bbox_path[i].split('/')[1] == 't2w_ax_0':
            origin_size = int(sizes[1])
        else:
            assert cancer_bbox_path[i].split('/')[1] == 't2wfs_ax_0'
            origin_size = int(sizes[2])

        file_path = os.path.join(label_path, cancer_bbox_path[i])
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

        if cancer_bbox_path[i].split('/')[1] != 'dwi_ax_0':
            resize_factor = (np.array((dwi_size, dwi_size)).astype(np.float) + 1e-3) / bboxes_image.shape  # +1e-3 to suppress warning of scipy.ndimage.zoom
            bboxes_image = ndimage.interpolation.zoom(bboxes_image.astype(np.float), resize_factor, order=0)

        if i == 0:
            bboxes = bboxes_image
        else:
            bboxes = np.logical_or(bboxes, bboxes_image)

    bboxes = np.expand_dims(bboxes, 0)
    print(type(bboxes[0, 0]))
    print(bboxes.shape)
    tl.visualize.save_images(bboxes, [1, 1], '/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/label_visualize.png')
    
    #print(np.sum(np.where(bboxes.any() != 1 or bboxes.any() != 0)))
    '''
    '''
    path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/period/all_sizes_periods.csv'
    accession_list = []
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        for data in reader:

            accession = data[0].split('/')[0]
            if accession not in accession_list:
                accession_list.append(accession)

    print(len(accession_list))
    '''
    '''
    path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/period/all_sizes_periods.csv'
    accession_list = []
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        for data in reader:

            accession = data[0].split('/')[0]
            if accession not in accession_list:
                accession_list.append(accession)

    print(len(accession_list))
    '''
    '''
##############################################review steps of generating masks##################################
    labels_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels"
    dcm_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series"
    record_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv"

    accession_labeled = os.listdir(labels_path)

    i = 0
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)

        for accession in reader:

            accession = accession[0]
            if accession not in accession_labeled:
                continue
            if i < 4:
                i += 1
                continue
            print(accession)
            accession_path = os.path.join(dcm_path, accession)

            mode_path = os.path.join(accession_path, "dwi_ax_0")

            dwi_ordered_dict, spacing = read_dwi(mode_path)
            b_values = np.array(dwi_ordered_dict.keys())
            nearest_b_low = preprocess_util.find_nearest(b_values, B_LOW)
            nearest_b_high = preprocess_util.find_nearest(b_values, B_HIGH)
            assert np.abs(nearest_b_low - B_LOW) <= 500
            assert np.abs(nearest_b_high - B_HIGH) <= 500
            tmp_dwi_dict = {nearest_b_low: dwi_ordered_dict[nearest_b_low], nearest_b_high: dwi_ordered_dict[nearest_b_high]}
            image = preprocess_util.dwi2adc(tmp_dwi_dict)
            print(image.shape)
            if nearest_b_low == B_LOW:
                dwi_low = dwi_ordered_dict[B_LOW]
            else:
                dwi_low = preprocess_util.calculate_dwi(image, dwi_ordered_dict[nearest_b_low], nearest_b_low, B_LOW)

            if nearest_b_high == B_HIGH:
                dwi_high = dwi_ordered_dict[B_HIGH]
            else:
                dwi_high = preprocess_util.calculate_dwi(image, dwi_ordered_dict[nearest_b_high], nearest_b_high, B_HIGH)
            assert dwi_low.shape == image.shape
            assert dwi_high.shape == image.shape

            image[np.isinf(image) | np.isnan(image) | (image < 0)] = 0
            resampled_adc, new_spacing = preprocess_util.resample(image, spacing, NEW_SPACING)
            resampled_dwi_low = preprocess_util.resize(dwi_low, resampled_adc.shape)

            voxel_volumn = 1
            for sp in new_spacing:
                voxel_volumn *= sp

            print(resampled_adc.shape)
            def get_bladder_mask(img, scale):

                denoised = preprocess_util.denoise(img.astype(np.float))
                threshold = preprocess_util.get_bladder_threshold(denoised, voxel_volumn, MIN_BLADDER_VOLUMN, scale)

                plt.hist(denoised.flatten(), bins=200)
                plt.show()
                plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/hist.png')

                return denoised > threshold

            mask_adc = get_bladder_mask(resampled_adc, ADC_THRESHOLD_SCALE)
            mask_dwi_low = get_bladder_mask(resampled_dwi_low, DWI_LOW_THRESHOLD_SCALE)
            mask = np.logical_and(mask_adc, mask_dwi_low)
            mask = preprocess_util.postprocess_bladder_mask(mask)

            dilated_mask = preprocess_util.get_dilated_mask(mask, DILATED_PIXELS)
            dilated_mask = preprocess_util.resize(dilated_mask.astype(np.float), image.shape)
            dilated_mask = dilated_mask > 0.5

            break
    '''
    '''
######################################accessions with period#####################################
    record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/period/0_period_train.csv'
    record_path1 = '/DB/rhome/qyzheng/Desktop/Link to renji_data/labels/bladder_tags_period.csv'

    with open(record_path1, 'rb') as f:
        reader = list(csv.reader(f))
        accessions = zip(*reader)[0]

    with open(record_path, 'rb') as f:
        reader = list(csv.reader(f))
        accession = [s.split('/')[0] for s in zip(*reader)[0]]
        i = 0
        if i == 0:
            print np.array(reader)[0, 9]
            i += 1

    print len(set(accessions))
    print len(set(accession))
    '''
    '''
#######################################label of period########################################
    path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/labels/bladder_tags_period.csv'
    record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/1_cv_train.csv'
    
    periods = {}
    labels = {'0': 0, '1': 0, '2': 0}
    with open(path, 'rb') as f:
        reader = csv.reader(f)

        for item in reader:

            accession = item[0]
            period = item[1]
            if not periods.has_key(period):
                periods[period] = {'sum': 0, 'accession': [accession]}
            else:
                periods[period]['accession'].append(accession)


    with open(record_path, 'rb') as f:
        reader = csv.reader(f)

        for item in reader:

            accession = item[0].split('/')[0]
            label = item[2]
            for period in periods.keys():

                if accession in periods[period]['accession']:
                    periods[period]['sum'] += 1
                    labels[label] += 1

    for key, value in periods.items():

        print key, value['sum']

    for key, value in labels.items():

        print key, value
    '''
    '''
########################################check the dwi_t2w_t2wfs_equal.csv#############################
    path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series'
    save_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv'
    save_path1 = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs.csv'
    labels_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels"

    num_dwi_t2w_t2wfs = 0
    num_equal = 0
    labeled_list = os.listdir(labels_path)
    mode_target = ['dwi_ax_0', 't2w_ax_0', 't2wfs_ax_0']
    with open(save_path, 'wb') as f:
        writer = csv.writer(f)

        with open(save_path1, 'wb') as f1:
            writer1 = csv.writer(f1)

            for accession in os.listdir(path):

                if accession not in labeled_list:
                    continue
                accession_path = os.path.join(path, accession)
                mode_list = os.listdir(accession_path)

                if set(mode_target).issubset(mode_list):
                    num_dcm = []
                    num_dwi_t2w_t2wfs += 1
                    writer1.writerow([accession])

                    for mode in mode_target:

                        mode_path = os.path.join(accession_path, mode)
                        stack_path = os.path.join(mode_path, os.listdir(mode_path)[0])
                        num_dcm.append(len([s for s in os.listdir(stack_path) if s.endswith('.dcm')]))

                    if len(set(num_dcm)) == 1:
                        num_equal += 1
                        writer.writerow([accession])

    print num_dwi_t2w_t2wfs
    print num_equal

    with open(save_path, 'rb') as f:
        print len(list(csv.reader(f)))

    with open(save_path1, 'rb') as f:
        print len(list(csv.reader(f)))
    '''
    '''
#######################################explore the origin data###############################
    record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs.csv'
    path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series'
    save_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv'

    i = 0
    min_slice = 10000
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)
        with open(save_path, 'rb') as f1:
            reader1 = csv.reader(f1)

            for accession in reader:

                if accession not in reader1:
                    print accession[0]
                    dcm = {"dwi_ax_0":0, "t2w_ax_0":0, "t2wfs_ax_0":0}
                    stacks = {"dwi_ax_0":0, "t2w_ax_0":0, "t2wfs_ax_0":0}

                    accession_path = os.path.join(path, accession[0])
                    for mode in os.listdir(accession_path):

                        if mode == "t2w_ax_0" or mode == "t2wfs_ax_0" or mode == "dwi_ax_0":
                            mode_path = os.path.join(accession_path, mode)
                            stack = os.listdir(mode_path)
                            stacks[mode] = os.path.join(mode_path, stack[0])
                            dcm[mode] = os.listdir(stacks[mode])
                            if min_slice > len(dcm[mode]) - 1:
                                min_slice = len(dcm[mode]) - 1
                        else:
                            continue

                    dwi_path = os.path.join(stacks["dwi_ax_0"], dcm["dwi_ax_0"][min_slice])
                    t2w_path = os.path.join(stacks["t2w_ax_0"], dcm["t2w_ax_0"][min_slice])
                    t2wfs_path = os.path.join(stacks["t2wfs_ax_0"], dcm["t2wfs_ax_0"][min_slice])

                    dwi = dicom.read_file(dwi_path).ImagePositionPatient
                    t2w = dicom.read_file(t2w_path).ImagePositionPatient
                    t2wfs = dicom.read_file(t2wfs_path).ImagePositionPatient

                    print 'dwi_%d: %r' %(min_slice, dwi)
                    print 't2w_%d: %r' %(min_slice, t2w)
                    print 't2wfs_%d: %r' %(min_slice, t2wfs)

                    if len(dcm["dwi_ax_0"]) - 1 > min_slice:
                        dwi_path = os.path.join(stacks["dwi_ax_0"], dcm["dwi_ax_0"][min_slice+1])
                        dwi = dicom.read_file(dwi_path).ImagePositionPatient
                        print 'next dwi_%d: %r' %(min_slice+1, dwi)

                    if len(dcm["t2w_ax_0"]) - 1 > min_slice:
                        t2w_path = os.path.join(stacks["t2w_ax_0"], dcm["t2w_ax_0"][min_slice+1])
                        t2w = dicom.read_file(t2w_path).ImagePositionPatient
                        print 'next t2w_%d: %r' %(min_slice+1, t2w)

                    if len(dcm["t2wfs_ax_0"]) - 1 > min_slice:
                        t2wfs_path = os.path.join(stacks["t2wfs_ax_0"], dcm["t2wfs_ax_0"][min_slice+1])
                        t2wfs = dicom.read_file(t2wfs_path).ImagePositionPatient
                        print 'next t2wfs_%d: %r' %(min_slice+1, t2wfs)

                    if i < 3:
                        i += 1
                    else:
                        break                   
    '''
    '''
########################################csv for mask rcnn#############################################
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_train.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_val.csv'
    test_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/test.csv'
    save_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/mask_rcnn'

    num_train = 0
    num_test = 0

    with open(os.path.join(save_path, 'train.csv'), 'wb') as f:
        writer = csv.writer(f)
        with open(train_path, 'rb') as f1:
            reader1 = csv.reader(f1)
            for i in reader1:

                if i[3] == '1':
                    writer.writerow(i)
                    num_train += 1
        with open(val_path, 'rb') as f2:
            reader2 = csv.reader(f2)
            for i in reader2:

                if i[3] == '1':
                    writer.writerow(i)
                    num_train += 1

    with open(os.path.join(save_path, 'test.csv'), 'wb') as f3:
        writer3 = csv.writer(f3)
        with open(test_path, 'rb') as f4:
            reader4 = csv.reader(f4)
            for i in reader4:

                if i[3] == '1':
                    writer3.writerow(i)
                    num_test += 1

    print(num_train)
    print(num_test)
    '''
    '''
#####################################merge train.csv and val.csv###############################
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/1_0/0_label_1_0_train.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/1_0/0_label_1_0_val.csv'
    save_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/1_0/0_label_1_0_train_val.csv'

    num = 0
    with open(save_path, 'wb') as f:
        writer = csv.writer(f)
        with open(train_path, 'rb') as f1:
            reader = csv.reader(f1)
            for i in reader:

                writer.writerow(i)
                num += 1
        with open(val_path, 'rb') as f2:
            reader1 = csv.reader(f2)
            for i in reader1:

                writer.writerow(i)
                num += 1
    print(num)

    with open(save_path, 'rb') as f3:
        reader2 = csv.reader(f3)
        print(len(list(reader2)))

    '''
    '''
##################################empty cancer bbox?################################
    path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels/D0564603/dwi_ax_0/stack0_b0guess/box_label_1.txt'
    with open(path, 'rb') as f:
        cancer_bbox = pickle.load(f)
    assert cancer_bbox == []
    print(cancer_bbox)
    print(type(cancer_bbox))
    '''
    '''
#######################################csv whose label is 1 and 0####################################
    record_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/all_sizes.csv"
    save_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/all_label_1_0.csv"

    num = 0
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)
        with open(save_path, 'wb') as f1:
            writer = csv.writer(f1)
            for row in reader:

                if row[2] != '2':
                    writer.writerow(row)
                    num += 1
                else:
                    continue

    print(num)
    '''
    '''
###########################################size of dwi t2w tw2fs############################
    dcm_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series"
    record_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv"

    equal = 0
    twice = 0
    num = 0
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)
        for accession_ls in reader:

            num += 1
            accession = accession_ls[0]
            accession_path = os.path.join(dcm_path, accession)
            mode_list = os.listdir(accession_path)
            if num % 10 == 0:
                print(num)
            if "t2w_ax_0" in mode_list and "t2wfs_ax_0" in mode_list and "dwi_ax_0" in mode_list:

                dwi_path = os.path.join(accession_path, "dwi_ax_0")
                dilated_mask, dwi_image = preprocess_dwi(dwi_path)
                t2w_path = os.path.join(accession_path, "t2w_ax_0")
                t2wfs_path = os.path.join(accession_path, "t2wfs_ax_0")
                t2w_image = preprocess_others(t2w_path)
                t2wfs_image = preprocess_others(t2wfs_path)
                if t2w_image.shape == t2wfs_image.shape:
                    equal += 1
                    if dilated_mask.shape[0] * 2 == t2wfs_image.shape[0]:
                        twice += 1
            else:
                print(accession)
                continue

    print(num)
    print(equal)
    print(twice)

    #478
    #360
    #274

    '''
###########################################for csv######################################
    '''
    record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv'
    path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series'
    save_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal_pos.csv'

    #num = 3
    num_1 = 0
    num_2 = 0
    num_3 = 0
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)
        with open(save_path, 'wb') as f1:
            writer = csv.writer(f1)
            for accession in reader:

                num_3 += 1
                ImagePosition = {"t2w_ax_0":None, "t2wfs_ax_0":None, "dwi_ax_0":None}
                accession_path = os.path.join(path, accession[0])
                for mode in os.listdir(accession_path):

                    if mode == "t2w_ax_0" or mode == "t2wfs_ax_0" or mode == "dwi_ax_0":
                        mode_path = os.path.join(accession_path, mode)
                        stack = os.listdir(mode_path)
                        stack_path = os.path.join(mode_path, stack[0])
                        dcm = os.listdir(stack_path)
                        dcm_path = os.path.join(stack_path, dcm[0])
                        ImagePosition[mode] = list(map(float,dicom.read_file(dcm_path).ImagePositionPatient))
                    else:
                        continue

                if same_position(ImagePosition["dwi_ax_0"], ImagePosition["t2w_ax_0"], 3) and same_position(ImagePosition["dwi_ax_0"], ImagePosition["t2wfs_ax_0"], 3):
                    num_1 += 1
                if same_position(ImagePosition["dwi_ax_0"], ImagePosition["t2w_ax_0"], 2) and same_position(ImagePosition["dwi_ax_0"], ImagePosition["t2wfs_ax_0"], 2):
                    writer.writerow([accession[0]])
                    num_2 += 1
    print(num_1)
    print(num_2)
    print(num_3)
    '''
    '''
    record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs.csv'
    path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series'
    save_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv'

    num = 0
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)
        with open(save_path, 'wb') as f1:
            writer = csv.writer(f1)

            num_acc = {"equal":0, "t2w==t2wfs":0, "dwi==t2w":0, "dwi==t2wfs":0}
            position = {"equal":0, "t2w==t2wfs":0, "dwi==t2w":0}
            for accession in reader:

                num += 1
                num_dcm = {"t2w_ax_0":0, "t2wfs_ax_0":0, "dwi_ax_0":0}
                ImagePosition = {"t2w_ax_0":None, "t2wfs_ax_0":None, "dwi_ax_0":None}
                accession_path = os.path.join(path, accession[0])
                for mode in os.listdir(accession_path):

                    if mode == "t2w_ax_0" or mode == "t2wfs_ax_0" or mode == "dwi_ax_0":
                        mode_path = os.path.join(accession_path, mode)
                        stack = os.listdir(mode_path)
                        stack_path = os.path.join(mode_path, stack[0])
                        dcm = os.listdir(stack_path)
                        dcm_path = os.path.join(stack_path, dcm[0])
                        ImagePosition[mode] = dicom.read_file(dcm_path).ImagePositionPatient
                        num_dcm[mode] = len(dcm)
                    else:
                        continue
                if num_dcm["t2w_ax_0"] == num_dcm["t2wfs_ax_0"]:
                    num_acc["t2w==t2wfs"] += 1
                    if num_dcm["t2w_ax_0"] == num_dcm["dwi_ax_0"]:
                        num_acc["equal"] += 1
                        writer.writerow([accession[0]])
                        if ImagePosition["t2w_ax_0"] == ImagePosition["t2wfs_ax_0"]:
                            position["t2w==t2wfs"] += 1
                            if ImagePosition["t2w_ax_0"] == ImagePosition["dwi_ax_0"]:
                                position["equal"] += 1
                        if ImagePosition["t2w_ax_0"] == ImagePosition["dwi_ax_0"]:
                            position["dwi==t2w"] += 1
                if num_dcm["t2w_ax_0"] == num_dcm["dwi_ax_0"]:
                    num_acc["dwi==t2w"] += 1
                if num_dcm["t2wfs_ax_0"] == num_dcm["dwi_ax_0"]:
                    num_acc["dwi==t2wfs"] += 1

            for key, value in num_acc.items():
                print(key, value)
            for key, value in position.items():
                print(key, value)
    
    print(num)
    '''
    '''
    labels_path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels'
    record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process'
    path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series"

    accession_list = os.listdir(labels_path)
    mode_dict = {}
    stack_max = 0
    stack_path = None
    number = {"dwi_t2w":0, "dwi_t2wfs":0, "dwi_t2w_t2wfs":0, "dwi_t2w_t2wfs_dce":0}

    f1 = open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w.csv', 'ab')
    writer1 = csv.writer(f1)
    f2 = open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2wfs.csv', 'ab')
    writer2 = csv.writer(f2)
    f3 = open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs.csv', 'ab')
    writer3 = csv.writer(f3)
    f4 = open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_dce.csv', 'ab')
    writer4 = csv.writer(f4)

    for accession in os.listdir(path):

        if accession not in accession_list:
            continue
        accession_path = os.path.join(path, accession)
        label_path = os.path.join(labels_path, accession)
        label_mode_list = os.listdir(label_path)
        mode_list = os.listdir(accession_path)

        if "dwi_ax_0" in mode_list and "t2w_ax_0" in mode_list and "dwi_ax_0" in label_mode_list and "t2w_ax_0" in label_mode_list:
            writer1.writerow([accession])
            number["dwi_t2w"] += 1
        if "dwi_ax_0" in mode_list and "t2wfs_ax_0" in mode_list and "dwi_ax_0" in label_mode_list and "t2wfs_ax_0" in label_mode_list:
            writer2.writerow([accession])
            number["dwi_t2wfs"] += 1
        if "dwi_ax_0" in mode_list and "t2w_ax_0" in mode_list and "t2wfs_ax_0" in mode_list and "dwi_ax_0" in label_mode_list and "t2w_ax_0" in label_mode_list and "t2wfs_ax_0" in label_mode_list:
            writer3.writerow([accession])
            number["dwi_t2w_t2wfs"] += 1
        if "dwi_ax_0" in mode_list and "t2w_ax_0" in mode_list and "t2wfs_ax_0" in mode_list and "dce_ax_0" in mode_list and "dwi_ax_0" in label_mode_list and "t2w_ax_0" in label_mode_list and "t2wfs_ax_0" in label_mode_list and "dce_ax_0" in label_mode_list:
            writer4.writerow([accession])
            number["dwi_t2w_t2wfs_dce"] += 1

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    
    for key, value in number.items():
        print(key, value)
    '''
    '''
		for mode in mode_list:

			if not mode_dict.has_key(mode):
				mode_dict[mode] = {'sum':1}
			else:
				mode_dict[mode]['sum'] += 1
			mode_path = os.path.join(accession_path, mode)

			stack_list = os.listdir(mode_path)
			if stack_max <= len(stack_list):
				stack_max = len(stack_list)
				stack_path = mode_path
			#assert len(stack_list) <= 4
			#mode_dict[mode]['stack'][(len(stack_list)-1)] += 1

	print("total accession: ", len(accession_list))
	for key, value in number.items():

		print(key, value)
	print("max stack: ", stack_max, "path:", stack_path)
	'''
    '''
	('total accession: ', 785)
	('dwi_t2w_t2wfs', 677)
	('all', 121)
	('dwi_t2w', 696)
	('dwi_t2wfs', 755)

	('dce_ax_0', {'sum': 169})
	('dwi_ax_0', {'sum': 775})
	('t2w_sag_0', {'sum': 445})
	('t2wfs_ax_0', {'sum': 756})
	('t2wfs_cor_0', {'sum': 332})
	('t2w_ax_0', {'sum': 699})
	'''
    '''
    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_label_train.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_label_val.csv'
    train = []
    val = []
    train_size = 0
    val_size = 0
    
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            train_size += 1

    with open(val_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            val_size += 1

    print(train_size)
    print(val_size)
	'''
    '''
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            label = i[3]
            if label == "1" or label == "2": 
                train.append(i)
                train_size += 1

    with open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_label_train.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(train)

    with open(val_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:

            label = i[3]
            if label == "1" or label == "2": 
                val.append(i)
                val_size += 1

    with open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_label_val.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(val)
	'''
    #image, label = next_batch(train, 1, 160, 160, 1000)
    '''
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    a = plt.subplot(1, 1, 1)

    max_shape = 0
    x = list(range(1, 441))
    shape = [0] * 440
    for i in range(train_size):

        image = np.load(os.path.join(senior_path, train[i][0]))
        shape[image[0].shape[0]-1] += 1

    for i in range(val_size):

        image = np.load(os.path.join(senior_path, val[i][0]))
        shape[image[0].shape[0]-1] += 1

    plt.bar(x, shape)
    plt.show()
    plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/show/statistics.png')

    print(max_shape)
	'''

if __name__ == "__main__":

    main()