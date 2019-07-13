import pydicom as dicom
import os
import re
import numpy as np
import collections


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
    b_is_guessed = False
    for stack_dir in os.listdir(dwi_series_path):
        [_, b_value_str] = stack_dir.split('_')
        if 'None' not in b_value_str:
            if "guess" in b_value_str:
                b_is_guessed = True
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
    return dwi_ordered_dict, dwi_spacing, b_is_guessed

if __name__ == '__main__':
    import plot_3D_image, matplotlib.pyplot as plt
    dwi_ordered_dict, dwi_spacing, b_is_guessed = read_dwi('/DATA3_DB7/data/public/renji_data/bladder_cleaned_distinct_series/W0186949/dwi_ax_0')
    print dwi_ordered_dict
    fig = plt.figure()
    plane = plot_3D_image.Multi3DArrayPlane(fig, 1, 1)
    plane.add(dwi_ordered_dict[1000].transpose(2,0,1), cmap='gray', fixed_window=False)
    plane.ready()
    plt.show()