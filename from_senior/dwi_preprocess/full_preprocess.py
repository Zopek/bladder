import read_dwi
import numpy as np
import preprocess_util
import os
import traceback
import json
import multiprocessing
import sys

B_LOW = 0
B_HIGH = 1000
NEW_SPACING_SIZE = 1.25
NEW_SPACING = np.array([NEW_SPACING_SIZE, NEW_SPACING_SIZE, NEW_SPACING_SIZE])  # unit is mm
MIN_BLADDER_VOLUMN = 25000  # unit is mm^3

ADC_THRESHOLD_SCALE = 0.7
DWI_LOW_THRESHOLD_SCALE = 0.4

DILATED_PIXELS = int(np.round(10 / NEW_SPACING_SIZE))


def full_preprocess(series_path, use_origin_spacing):
    # get dwi_b0, dwi_b1000 and adc
    dwi_ordered_dict, spacing, _ = read_dwi.read_dwi(series_path)
    b_values = np.array(dwi_ordered_dict.keys())
    nearest_b_low = preprocess_util.find_nearest(b_values, B_LOW)
    nearest_b_high = preprocess_util.find_nearest(b_values, B_HIGH)
    assert np.abs(nearest_b_low - B_LOW) <= 500
    assert np.abs(nearest_b_high - B_HIGH) <= 500
    tmp_dwi_dict = {nearest_b_low: dwi_ordered_dict[nearest_b_low], nearest_b_high: dwi_ordered_dict[nearest_b_high]}
    adc = preprocess_util.dwi2adc(tmp_dwi_dict)
    if nearest_b_low == B_LOW:
        dwi_low = dwi_ordered_dict[B_LOW]
    else:
        dwi_low = preprocess_util.calculate_dwi(adc, dwi_ordered_dict[nearest_b_low], nearest_b_low, B_LOW)

    if nearest_b_high == B_HIGH:
        dwi_high = dwi_ordered_dict[B_HIGH]
    else:
        dwi_high = preprocess_util.calculate_dwi(adc, dwi_ordered_dict[nearest_b_high], nearest_b_high, B_HIGH)
    assert dwi_low.shape == adc.shape
    assert dwi_high.shape == adc.shape

    # get bladder mask
    adc[np.isinf(adc) | np.isnan(adc) | (adc < 0)] = 0
    resampled_adc, new_spacing = preprocess_util.resample(adc, spacing, NEW_SPACING)
    resampled_dwi_low = preprocess_util.resize(dwi_low, resampled_adc.shape)

    voxel_volumn = 1
    for sp in new_spacing:
        voxel_volumn *= sp

    def get_bladder_mask(image, scale):
        denoised = preprocess_util.denoise(image.astype(np.float))
        threshold = preprocess_util.get_bladder_threshold(denoised, voxel_volumn, MIN_BLADDER_VOLUMN, scale)
        return denoised > threshold

    mask_adc = get_bladder_mask(resampled_adc, ADC_THRESHOLD_SCALE)
    mask_dwi_low = get_bladder_mask(resampled_dwi_low, DWI_LOW_THRESHOLD_SCALE)
    mask = np.logical_and(mask_adc, mask_dwi_low)

    mask = preprocess_util.postprocess_bladder_mask(mask)
    dilated_mask = preprocess_util.get_dilated_mask(mask, DILATED_PIXELS)

    # print 'nearest_b_low:', nearest_b_low, 'nearest_b_high:', nearest_b_high
    # print 'bladder_volumn:', np.sum(mask) * voxel_volumn

    if use_origin_spacing:
        mask = preprocess_util.resize(mask.astype(np.float), adc.shape)
        mask = mask > 0.5
        dilated_mask = preprocess_util.resize(dilated_mask.astype(np.float), adc.shape)
        dilated_mask = dilated_mask > 0.5

        whole_adc = adc
        whole_dwi_low = dwi_low
        whole_dwi_high = dwi_high
    else:
        whole_adc = resampled_adc
        whole_dwi_low = resampled_dwi_low
        whole_dwi_high = preprocess_util.resize(dwi_high, resampled_adc.shape)
    return whole_adc, whole_dwi_low, whole_dwi_high, mask.astype(np.bool), dilated_mask.astype(np.bool)


def slices2lists(slices_list):
    return [[s.start, s.stop, s.step] for s in slices_list]


def process_work(series_path, target_prefix, print_lock):
    try:
        whole_adc, whole_dwi_b0, whole_dwi_b1000, mask, dilated_mask = full_preprocess(series_path, True)
        image = np.stack([whole_adc, whole_dwi_b0, whole_dwi_b1000]).astype(np.float32)
        os.makedirs(target_prefix)
        num_slice = image.shape[-1]
        for i in range(num_slice):
            np.save(os.path.join(target_prefix, 'image_{}.npy'.format(i)), image[:, :, :, i])
            np.save(os.path.join(target_prefix, 'mask_{}.npy'.format(i)), mask[:, :, i])
            np.save(os.path.join(target_prefix, 'dilated_mask_{}.npy'.format(i)), dilated_mask[:, :, i])
        mask_bbox = preprocess_util.find_bounding_box(mask)
        dilated_mask_bbox = preprocess_util.find_bounding_box(dilated_mask)
        with open(os.path.join(target_prefix, 'mask_bbox.json'), 'wb') as fp:
            json.dump(slices2lists(mask_bbox), fp)
        with open(os.path.join(target_prefix, 'dilated_mask_bbox.json'), 'wb') as fp:
            json.dump(slices2lists(dilated_mask_bbox), fp)
        print_lock.acquire()
        print 'Succ: {}'.format(series_path)
        print_lock.release()
        return 0
    except:
        trace_str = traceback.format_exc()
        print_lock.acquire()
        print 'Error: {}'.format(series_path)
        print trace_str
        print_lock.release()
        return 1

def main(root_path, target_root_path, num_processes):
    # args_list = []
    result_list = []
    pool = multiprocessing.Pool(int(num_processes))
    lock = multiprocessing.Manager().Lock()

    for accession_dir in os.listdir(root_path):
        accession_path = os.path.join(root_path, accession_dir)
        if not os.path.isdir(accession_path):
            continue
        for series_dir in os.listdir(accession_path):
            # print series_dir
            if 'dwi_ax' not in series_dir:
                continue
            series_path = os.path.join(accession_path, series_dir)
            if not os.path.isdir(series_path):
                continue
            target_prefix = os.path.join(target_root_path, accession_dir, series_dir)
            args = (series_path, target_prefix, lock,)
            r = pool.apply_async(process_work, args)
            result_list.append(r)
            # args_list.append((series_path, target_prefix, lock,))
            # lock.acquire()
            # print 'Starting: {}'.format(series_path)
            # lock.release()
    # pool.map(process_work, args_list)
    pool.close()
    pool.join()
    num_succ = 0
    num_fail = 0
    for r in result_list:
        if r.get() == 0:
            num_succ += 1
        else:
            num_fail += 1
    print "Finish: num_succ = {}; num_fail = {}".format(num_succ, num_fail)


if __name__ == '__main__':
    # root = '/DATA3_DB7/data/public/renji_data/bladder_cleaned_distinct_series'
    # target = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d'
    root = sys.argv[1]
    target = sys.argv[2]
    num_parallel = sys.argv[3]
    main(root, target, num_parallel)
