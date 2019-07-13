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

    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor) + 1e-3
    real_resize_factor = new_shape / image.shape
    real_new_spacing = spacing / real_resize_factor
    new_image = ndimage.interpolation.zoom(image, real_resize_factor, order=2)

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

    return dilated_mask, image, dwi_low, dwi_high

def preprocess_others(mode_path):

    image, spacing = read_dcm(mode_path)
    image = image[0]
    # only dwi and dce have more than one stacks

    image[np.isinf(image) | np.isnan(image) | (image < 0)] = 0

    return image

def find_bounding_box(a):
    objects = list(ndimage.find_objects(a))
    assert len(objects) == 1
    return objects[0]

def slices2lists(slices_list):
    return [[s.start, s.stop, s.step] for s in slices_list]
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
save_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal"

accession_labeled = os.listdir(labels_path)
accession_processed = os.listdir(save_path)

with open(record_path, 'rb') as f:
    reader = csv.reader(f)

    for accession in reader:

        accession = accession[0]
        if accession not in accession_labeled:
            continue
        if accession in accession_processed:
            continue
        accession_path = os.path.join(dcm_path, accession)

        dilated_mask, dwi_adc, dwi_low, dwi_high = preprocess_dwi(os.path.join(accession_path, "dwi_ax_0"))
        t2w_image = preprocess_others(os.path.join(accession_path, "t2w_ax_0"))
        t2wfs_image = preprocess_others(os.path.join(accession_path, "t2wfs_ax_0"))

        t2w_image = preprocess_util.resize(t2w_image.astype(np.float), dwi_adc.shape)
        t2wfs_image = preprocess_util.resize(t2wfs_image.astype(np.float), dwi_adc.shape)
        image = np.stack([dwi_adc, dwi_low, dwi_high, t2w_image, t2wfs_image]).astype(np.float32)

        target_path = os.path.join(save_path, accession)
        os.makedirs(target_path)

        for i in range(image.shape[-1]):
            np.save(os.path.join(target_path, 'image_{}.npy'.format(i)), image[:, :, :, i])
        dilated_mask_bbox = find_bounding_box(dilated_mask)
        with open(os.path.join(target_path, 'dilated_mask_bbox.json'), 'wb') as fp:
            json.dump(slices2lists(dilated_mask_bbox), fp)

print len(os.listdir(save_path))

