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
    assert len(ds_list) > 1
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
'''
data_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series"

for accession in os.listdir(data_path):

	accession_path = os.path.join(data_path, accession)
	for mode in os.listdir(accession_path):

		mode_path = os.path.join(accession_path, mode)
		a = preprocess(mode_path)
'''
labels_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels"
dcm_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series"
record_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv"
record_path1 = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal_pos.csv"

num = 5

with open(record_path, 'rb') as f:
    reader = csv.reader(f)
    with open(record_path1, 'rb') as f1:
        reader1 = csv.reader(f1)
        for accession_ls in reader:

            if accession_ls in reader1:
                continue
            accession = "D1563734"
            if accession == "D0846177" or accession == "D1314058":
                continue
            accession_path = os.path.join(dcm_path, accession)
            mode_list = os.listdir(accession_path)
            if "t2w_ax_0" in mode_list and "t2wfs_ax_0" in mode_list and "dwi_ax_0" in mode_list:
                print(accession)
                dwi_path = os.path.join(accession_path, "dwi_ax_0")
                dilated_mask, dwi_image = preprocess_dwi(dwi_path)
                print(dilated_mask.shape)
                print(dwi_image.shape)
                t2w_path = os.path.join(accession_path, "t2w_ax_0")
                t2wfs_path = os.path.join(accession_path, "t2wfs_ax_0")
                t2w_image = preprocess_others(t2w_path)
                t2wfs_image = preprocess_others(t2wfs_path)

                label_list = []
                cancer_bbox_dict = {"dwi_ax_0":[], "t2w_ax_0":[], "t2wfs_ax_0":[]}
                for mode in os.listdir(accession_path):

                    if mode == "t2w_ax_0" or mode == "t2wfs_ax_0" or mode == "dwi_ax_0":
                        mode_path = os.path.join(accession_path, mode)
                        stack = os.listdir(mode_path)
                        stack_path = os.path.join(mode_path, stack[0])
                        label_path = os.path.join(labels_path, accession, mode, stack[0])
                        labels = pickle.load(open(os.path.join(label_path, 'label.txt'), 'r'))

                        for j in range(len(labels)):

                            cancer_bbox_path = os.path.join(label_path, 'box_label_{}.txt'.format(j))
                            #cancer_bbox = read_cancer_bbox(cancer_bbox_path, t2w_image.shape[0], t2w_image.shape[1])
                            
                            if mode == "dwi_ax_0":
                                cancer_bbox = read_cancer_bbox(cancer_bbox_path, dwi_image.shape[0], dwi_image.shape[1])
                            elif mode == "t2w_ax_0":
                                cancer_bbox = read_cancer_bbox(cancer_bbox_path, t2w_image.shape[0], t2w_image.shape[1])
                                resize_factor = (np.round(dwi_image.shape[:2]).astype(np.float) + 1e-3) / cancer_bbox.shape  # +1e-3 to suppress warning of scipy.ndimage.zoom
                                cancer_bbox = ndimage.interpolation.zoom(cancer_bbox.astype(np.float), resize_factor, order=0)
                                #cancer_bbox = preprocess_util.resize(cancer_bbox.astype(np.float), dwi_image.shape[:2])
                            elif mode == "t2wfs_ax_0":
                                cancer_bbox = read_cancer_bbox(cancer_bbox_path, t2wfs_image.shape[0], t2wfs_image.shape[1])
                                resize_factor = (np.round(dwi_image.shape[:2]).astype(np.float) + 1e-3) / cancer_bbox.shape  # +1e-3 to suppress warning of scipy.ndimage.zoom
                                cancer_bbox = ndimage.interpolation.zoom(cancer_bbox.astype(np.float), resize_factor, order=0)
                               
                            cancer_bbox_dict[mode].append(cancer_bbox)

                            if labels[j] == 0 or j in label_list:
                                continue
                            else:
                                label_list.append(j)
                    else:
                        continue

                t2w_image = preprocess_util.resize(t2w_image.astype(np.float), dwi_image.shape)
                t2wfs_image = preprocess_util.resize(t2wfs_image.astype(np.float), dwi_image.shape)
                print(t2w_image.shape)
                print(t2wfs_image.shape)
                dilated_mask_dwi = dwi_image * dilated_mask
                #dilated_mask = ndimage.interpolation.zoom(dilated_mask, 2, order=0)
                dilated_mask_t2w = t2w_image * dilated_mask
                dilated_mask_t2wfs = t2wfs_image * dilated_mask

                for j in range(len(label_list)):

                    cancer = cancer_bbox_dict["dwi_ax_0"][label_list[j]] + cancer_bbox_dict["t2w_ax_0"][label_list[j]] + cancer_bbox_dict["t2wfs_ax_0"][label_list[j]]
                    
                    vmin = np.min(dwi_image[:,:,label_list[j]])
                    vmax = np.max(dwi_image[:,:,label_list[j]])

                    fig = plt.figure()
                    ax = fig.add_subplot(3, 3, 1)
                    fig.colorbar(ax.imshow(dwi_image[:,:,label_list[j]], vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
                    ax = fig.add_subplot(3, 3, 2)
                    labeled_dwi = dilated_mask_dwi[:,:,label_list[j]]
                    fig.colorbar(ax.imshow(labeled_dwi, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
                    ax = fig.add_subplot(3, 3, 3)
                    labeled_dwi[cancer != 0] = 10000
                    fig.colorbar(ax.imshow(labeled_dwi, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)

                    vmin = np.min(t2w_image[:,:,label_list[j]])
                    vmax = np.max(t2w_image[:,:,label_list[j]])

                    ax = fig.add_subplot(3, 3, 4)
                    fig.colorbar(ax.imshow(t2w_image[:,:,label_list[j]], vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
                    ax = fig.add_subplot(3, 3, 5)
                    labeled_t2w = dilated_mask_t2w[:,:,label_list[j]]
                    fig.colorbar(ax.imshow(labeled_t2w, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
                    ax = fig.add_subplot(3, 3, 6)
                    labeled_t2w[cancer != 0] = 10000
                    fig.colorbar(ax.imshow(labeled_t2w, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)

                    vmin = np.min(t2wfs_image[:,:,label_list[j]])
                    vmax = np.max(t2wfs_image[:,:,label_list[j]])

                    ax = fig.add_subplot(3, 3, 7)
                    fig.colorbar(ax.imshow(t2wfs_image[:,:,label_list[j]], vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
                    ax = fig.add_subplot(3, 3, 8)
                    labeled_t2wfs = dilated_mask_t2wfs[:,:,label_list[j]]
                    fig.colorbar(ax.imshow(labeled_t2wfs, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
                    ax = fig.add_subplot(3, 3, 9)
                    labeled_t2wfs[cancer != 0] = 10000
                    fig.colorbar(ax.imshow(labeled_t2wfs, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)

                    fig.show()
                    plt.show()
                    plt.savefig('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/from_senior/dwi_preprocess/sample/dwi_t2w_t2wfs_label/sample_{}_{}.png'.format(accession, label_list[j]))
                    plt.close()

            num -= 1
            if num == 0:
                break

            break