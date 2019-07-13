import numpy as np
import os
import json
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
        break
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

    dcm_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series"
    data_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
    record_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv"
    read_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/all.csv"
    save_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/all_sizes.csv"

    equal = 0
    twice = 0
    num = 0
    count = 0
    with open(read_path, 'rb') as f2:
        reader2 = csv.reader(f2)
        samples = np.array(list(reader2))
        with open(record_path, 'rb') as f:
            reader = csv.reader(f)
            with open(save_path, 'wb') as f1:
                writer = csv.writer(f1)
                for accession_ls in reader:

                    accession = accession_ls[0]
                    accession_path = os.path.join(dcm_path, accession)
                    dwi_path = os.path.join(accession_path, "dwi_ax_0")
                    dilated_mask, dwi_image = preprocess_dwi(dwi_path)
                    t2w_path = os.path.join(accession_path, "t2w_ax_0")
                    t2wfs_path = os.path.join(accession_path, "t2wfs_ax_0")
                    t2w_image = preprocess_others(t2w_path)
                    t2wfs_image = preprocess_others(t2wfs_path)

                    assert dwi_image.shape[0] != 0
                    assert t2w_image.shape[0] != 0
                    assert t2wfs_image.shape[0] != 0

                    json_file = json.load(open(os.path.join(data_path, accession, 'dilated_mask_bbox.json'), 'r'))
                    start_slice = np.array(json_file)[2, 0]
                    end_slice = np.array(json_file)[2, 1]
                    for i in range(start_slice, end_slice):
                        slices = (samples[count, 0],
                            samples[count, 1],
                            samples[count, 2],
                            samples[count, 3],
                            samples[count, 4],
                            samples[count, 5],
                            dwi_image.shape[0],
                            t2w_image.shape[0],
                            t2wfs_image.shape[0])
                        count += 1
                        writer.writerow(slices)

                    num += 1
                    if num % 10 == 0:
                        print(num)

    print(num)
    print(equal)
    print(twice)

if __name__ == "__main__":

    main()