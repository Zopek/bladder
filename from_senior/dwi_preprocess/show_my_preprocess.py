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
import preprocess_util

NEW_SPACING_SIZE = 1.25
NEW_SPACING = np.array([NEW_SPACING_SIZE, NEW_SPACING_SIZE, NEW_SPACING_SIZE])  # unit is mm
MIN_BLADDER_VOLUMN = 25000  # unit is mm^3
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

def preprocess(mode_path):

	image, spacing = read_dcm(mode_path)

	image = image[0]
	image[np.isinf(image) | np.isnan(image) | (image < 0)] = 0
	resampled_image, new_spacing = resample(image, spacing, NEW_SPACING)

	return resampled_image, new_spacing, image

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

for i in range(10):

	accession = os.listdir(dcm_path)
	accession_path = os.path.join(dcm_path, accession[i])
	for mode in os.listdir(accession_path):

		if mode == "t2w_ax_0" or mode == "t2wfs_ax_0" or mode == "dwi_ax_0":
			mode_path = os.path.join(accession_path, mode)


			resampled_image, new_spacing, image = preprocess(mode_path)
			print(image.shape)

			voxel_volumn = 1
			for sp in new_spacing:
				voxel_volumn *= sp

			def get_bladder_mask(image, scale):
				denoised = preprocess_util.denoise(image.astype(np.float))
				threshold = preprocess_util.get_bladder_threshold(denoised, voxel_volumn, MIN_BLADDER_VOLUMN, scale)
				return denoised > threshold

			mask_image = get_bladder_mask(resampled_image, 0.4)
			mask = preprocess_util.postprocess_bladder_mask(mask_image)
			dilated_mask = preprocess_util.get_dilated_mask(mask, DILATED_PIXELS)
	 		dilated_mask = preprocess_util.resize(dilated_mask.astype(np.float), image.shape)
			dilated_mask = dilated_mask > 0.5

			dilated_masked_adc = image * dilated_mask



			stack = os.listdir(mode_path)
			stack_path = os.path.join(mode_path, stack[0])
			label_path = os.path.join(labels_path, accession[i], mode, stack[0])
			labels = pickle.load(open(os.path.join(label_path, 'label.txt'), 'r'))
			for j in range(len(labels)):

				if labels[j] == 0:
					continue

				a = image
				print(a.shape)
				cancer_bbox_path = os.path.join(label_path, 'box_label_{}.txt'.format(j))
				cancer_bbox = read_cancer_bbox(cancer_bbox_path, a.shape[0], a.shape[1])

				image1 = image[:,:,j]
				vmin = np.min(image1)
				vmax = np.max(image1)

				fig = plt.figure()
				ax = fig.add_subplot(2, 2, 1)
				fig.colorbar(ax.imshow(image1, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)

				image2 = dilated_masked_adc[:,:,j]
				#vmin = np.min(image2)
				#vmax = np.max(image2)
				#image2[cancer_bbox != 0] = 1000
				'''
				objects = list(ndimage.find_objects(image))
				print(objects)
				'''
				ax = fig.add_subplot(2, 2, 2)
				fig.colorbar(ax.imshow(image2, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)

				ax = fig.add_subplot(2, 2, 3)
				ax.imshow(cancer_bbox)
				fig.show()
				plt.show()
				plt.savefig('./sample/0.7/sample{}_{}_{}_{}.png'.format(i, accession[i], mode, j))
		else:
			continue