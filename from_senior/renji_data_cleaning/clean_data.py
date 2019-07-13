import dicom
import os
import traceback
import collections
import re
import csv
import numpy as np
from dicom_numpy.combine_slices import _validate_slices_form_uniform_grid as validateStack


class BlankAccessionNumberException(Exception):
    def __init__(self, err='AccessionNumber Tag is blank!'):
        Exception.__init__(self, err)


class IndexIsNotUniqueException(Exception):
    def __init__(self, err='Index is not unique!'):
        Exception.__init__(self, err)


class SeriesLengthDisagreeException(Exception):
    def __init__(self, err='Series length disagree!'):
        Exception.__init__(self, err)


class DicomFile:
    def __init__(self, file_path, dataset):
        self.file_path = file_path
        self.dataset = dataset

'''
def at_overlay_data_or_pixel_data(tag, VR, length):
    return tag == (0x6000, 0x3000) or tag == (0x7fe0, 0x0010)


def read_dcm_before_overlay_data_and_pixel_data(file_name):
    with open(file_name, 'rb') as fp:
        return dicom.filereader.read_partial(fp, at_overlay_data_or_pixel_data)
'''


def remove_blank_slice(df_list):
    new_df_list = []
    for df in df_list:
        if df.dataset.pixel_array.std() != 0.0:
            new_df_list.append(df)
    return new_df_list


def remove_duplicate_slice(df_list):
    slice_dict = dict()
    for df in df_list:
        idx = (tuple(df.dataset.ImagePositionPatient), int(df.dataset.InstanceNumber))
        if idx in slice_dict:
            # if (df.dataset.pixel_array != slice_dict[idx].dataset.pixel_array).any():
            if not np.array_equal(df.dataset.pixel_array, slice_dict[idx].dataset.pixel_array):
                raise IndexIsNotUniqueException
        else:
            slice_dict[idx] = df
    slice_sorted_od = collections.OrderedDict(sorted(slice_dict.items()))
    return slice_sorted_od


def split_stacks(slice_sorted_od):
    position_od = collections.OrderedDict()
    max_series_length = 0
    for ((position, instance_number), df) in slice_sorted_od.items():
        instance_list = position_od.setdefault(position, [])
        instance_list.append(df)
        if len(instance_list) > max_series_length:
            max_series_length = len(instance_list)

    # check series_length
    for position, instance_od in position_od.items():
        if len(instance_od) != max_series_length:
            del position_od[position]
            print 'Warning:', SeriesLengthDisagreeException()

    stack_list = map(list, zip(*position_od.values()))

    # validate stacks
    for stack in stack_list:
        slice_ds_list = [df.dataset for df in stack]
        validateStack(slice_ds_list)
    return stack_list


def link_series(series_name, stack_dict, target_dir_path):
    series_counter = 0
    target_series_path = os.path.join(target_dir_path, '{}_{}'.format(series_name, series_counter))
    while os.path.exists(target_series_path):
        series_counter += 1
        target_series_path = os.path.join(target_dir_path, '{}_{}'.format(series_name, series_counter))
    os.makedirs(target_series_path)

    # link images
    for stack_name, slice_list in stack_dict.items():
        target_stack_path = os.path.join(target_series_path, stack_name)
        os.makedirs(target_stack_path)
        for i in range(len(slice_list)):
            target_path = os.path.join(target_stack_path, 'I_{}.dcm'.format(i))
            os.link(slice_list[i].file_path, target_path)
        record('{}_num_slice_in_stack'.format(series_name), len(slice_list))
    record('{}_num_stack'.format(series_name), len(stack_dict))

global_recorders = {}


def record(name, value, recorders=global_recorders):
    recorder = recorders.setdefault(name, {})
    recorder[value] = recorder.get(value, 0) + 1


def print_results(recorders=global_recorders):
    for name, recorder in sorted(recorders.items()):
        sorted_recorder = sorted(recorder.iteritems(), key=lambda r: r[0])
        print 'Result: {}: {}'.format(name, sorted_recorder)


def parse_orientation(iop):
    plane = np.cross(iop[0:3], iop[3:6])
    plane = [abs(x) for x in plane]
    threshold15 = 0.966  # cos(15degree)
    threshold30 = 0.866  # cos(30degree)
    if plane[2] > threshold30:
        if plane[2] > threshold15:
            return 'ax'
        else:
            return 'oax'

    elif plane[1] > threshold30:
        if plane[1] > threshold15:
            return 'cor'
        else:
            return 'ocor'
    elif plane[0] > threshold30:
        if plane[0] > threshold15:
            return 'sag'
        else:
            return 'osag'
    else:
        return 'other'


def judge_orientation(stack_list):
    orientation = parse_orientation(stack_list[0][0].dataset.ImageOrientationPatient)
    if orientation == 'other':
        print 'Warning: Orientation is other: {}'.format(stack_list[0][0].dataset.ImageOrientationPatient)
    for i in range(1, len(stack_list)):
        if orientation != parse_orientation(stack_list[i][0].dataset.ImageOrientationPatient):
            print 'Warning: Orientation disagree between stacks!'
    return orientation


def clean_t2w(df_list, target_accession_path):
    stack_list = split_stacks(remove_duplicate_slice(remove_blank_slice(df_list)))
    orientation = judge_orientation(stack_list)
    if len(stack_list) > 1:
        print 'Warning: More than one stack!'
    stack_dict = dict()
    for i in range(len(stack_list)):
        stack_name = 'stack{}'.format(i)
        stack_dict[stack_name] = stack_list[i]
        if len(stack_list[i]) < 10:
            print 'Warning: length of stack is less than 10!'
    link_series('t2w_{}'.format(orientation), stack_dict, target_accession_path)


def clean_t2wfs(df_list, target_accession_path):
    stack_list = split_stacks(remove_duplicate_slice(remove_blank_slice(df_list)))
    orientation = judge_orientation(stack_list)
    if len(stack_list) > 1:
        print 'Warning: More than one stack!'
    stack_dict = dict()
    for i in range(len(stack_list)):
        stack_name = 'stack{}'.format(i)
        stack_dict[stack_name] = stack_list[i]
        if len(stack_list[i]) < 10:
            print 'Warning: length of stack is less than 10!'
    link_series('t2wfs_{}'.format(orientation), stack_dict, target_accession_path)

b_value_blacklist = [0, 120, 24, 17, 9]


def clean_dwi(df_list, target_accession_path):
    stack_list = split_stacks(remove_duplicate_slice(remove_blank_slice(df_list)))
    orientation = judge_orientation(stack_list)
    stack_dict = dict()
    # sort stacks by pixel value (high to low)
    tmp_list = []
    for stack in stack_list:
        mean = 0
        for df in stack:
            mean += df.dataset.pixel_array.mean()
        mean /= float(len(stack))
        tmp_list.append((mean, stack))
    stack_list = [v[1] for v in reversed(sorted(tmp_list))]
    # find stack name
    for i in range(len(stack_list)):
        b_value = None
        for df in stack_list[i]:
            if 'DiffusionBValue' in df.dataset:
                if b_value is None:
                    b_value = int(df.dataset.DiffusionBValue)
                elif b_value != int(df.dataset.DiffusionBValue):
                    print 'Warning: Multiple b value in a stack!'
        stack_name = 'stack{}_b{}'.format(i, b_value)
        # try to guess b value
        if b_value is None:
            if i == 0:
                b_value = 0
            elif df.dataset.SeriesDescription.lower().strip() == 'ax dwi b=250n':
                b_value = 250 * i
            else:
                numbers = re.findall(r'\d+', df.dataset.SeriesDescription)
                b_value_list = [int(n) for n in numbers if int(n) not in b_value_blacklist]
                if i - 1 < len(b_value_list):
                    b_value = b_value_list[i - 1]
            if b_value is not None:
                stack_name = 'stack{}_b{}guess'.format(i, b_value)
        if b_value is None:
            print 'Warning: Can not get or guess b_value!'
        stack_dict[stack_name] = stack_list[i]
        if len(stack_list[i]) < 10:
            print 'Warning: length of stack is less than 10!'
    link_series('dwi_{}'.format(orientation), stack_dict, target_accession_path)


def clean_dce(df_list, target_accession_path):
    stack_list = split_stacks(remove_duplicate_slice(remove_blank_slice(df_list)))
    orientation = judge_orientation(stack_list)
    # sort stacks by content time
    tmp_list = []
    for stack in stack_list:
        mean = 0
        for df in stack:
            mean += float(df.dataset.ContentTime)
        mean /= float(len(stack))
        tmp_list.append((mean, stack))
    stack_list = [v[1] for v in reversed(sorted(tmp_list))]
    stack_dict = dict()
    for i in range(len(stack_list)):
        stack_name = 'stack{}'.format(i)
        stack_dict[stack_name] = stack_list[i]
        if len(stack_list[i]) < 10:
            print 'Warning: length of stack is less than 10!'
    link_series('dce_{}'.format(orientation), stack_dict, target_accession_path)


def read_sd_list(file_path):
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
    sd_list = []
    for line in lines:
        sd = line.strip().lower()
        if sd:
            sd_list.append(sd)
    return sd_list


def read_an_set(file_path):
    an_set = set()
    with open(file_path, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            assert row[0] not in an_set
            an_set.add(row[0])
    return an_set


def main():
    source_root_path = '/DATA3_DB7/data/public/renji_data/bladder'
    base_target_root_path = '/DATA3_DB7/data/yjgu/working/renji_data/bladder_cleaned_distinct_series'
    target_root_path = base_target_root_path
    i = 0
    while os.path.exists(target_root_path):
        target_root_path = '{}_{}'.format(base_target_root_path, i)
        i += 1
    t2w_sd_list_path = 't2w_sd.txt'
    t2wfs_sd_list_path = 't2wfs_sd.txt'
    dwi_sd_list_path = 'dwi_sd.txt'
    dce_sd_list_path = 'dce_sd.txt'
    sd_dict = dict()
    sd_dict['t2w'] = read_sd_list(t2w_sd_list_path)
    sd_dict['t2wfs'] = read_sd_list(t2wfs_sd_list_path)
    sd_dict['dwi'] = read_sd_list(dwi_sd_list_path)
    sd_dict['dce'] = read_sd_list(dce_sd_list_path)
    an_whitelist = read_an_set('/DATA3_DB7/data/public/renji_data/labels/bladder_tags_class.csv')
    study_counter = 0
    t2w_counter = 0
    t2wfs_counter = 0
    dwi_counter = 0
    dce_counter = 0
    cleaned_series_uid_dict = dict()
    for year_dir in os.listdir(source_root_path):
        year_path = os.path.join(source_root_path, year_dir)
        if not os.path.isdir(year_path):
            continue

        for accession_dir in os.listdir(year_path):
            accession_path = os.path.join(year_path, accession_dir)
            if not os.path.isdir(accession_path):
                continue

            for study_dir in os.listdir(accession_path):
                study_path = os.path.join(accession_path, study_dir)
                if not os.path.isdir(study_path):
                    continue
                print 'Info: study {}: {}'.format(study_counter, study_path)
                study_counter += 1

                for series_dir in os.listdir(study_path):
                    series_path = os.path.join(study_path, series_dir)
                    if not os.path.isdir(series_path):
                        continue

                    df_list = []
                    is_t2w = False
                    is_t2wfs = False
                    is_dwi = False
                    is_dce = False
                    is_other = True
                    for file_name in sorted(os.listdir(series_path)):
                        if file_name == 'VERSION' or '.npy' in file_name:
                            continue
                        file_path = os.path.join(series_path, file_name)
                        try:
                            df = DicomFile(file_path, dicom.read_file(file_path))
                            # ds = read_dcm_before_overlay_data_and_pixel_data(file_path)
                            if len(df_list) == 0:
                                an = df.dataset.AccessionNumber
                                if an not in an_whitelist:
                                    break
                                sd = df.dataset.SeriesDescription
                                sd = sd.strip().lower()
                                # print 'Info: ', file_path, sd
                                is_t2w = sd in sd_dict['t2w']
                                is_t2wfs = sd in sd_dict['t2wfs']
                                is_dwi = sd in sd_dict['dwi']
                                is_dce = sd in sd_dict['dce']
                                is_other = not(is_t2w or is_t2wfs or is_dwi or is_dce)
                                if is_other:
                                    break
                                series_uid = df.dataset.SeriesInstanceUID
                                if series_uid in cleaned_series_uid_dict:
                                    print 'Warning: Duplicate series! {} and {}'\
                                        .format(file_path, cleaned_series_uid_dict[series_uid])
                                    break
                            df_list.append(df)
                        except Exception, e:
                            print 'Error: Fail to read file: {}, msg: {}'.format(file_path, e.message)
                            continue
                    if not is_other and len(df_list) > 0:
                        try:
                            an = df_list[0].dataset.AccessionNumber
                            sd = df_list[0].dataset.SeriesDescription
                            series_uid = df_list[0].dataset.SeriesInstanceUID
                            for df in df_list:
                                assert df.dataset.AccessionNumber == an
                                assert df.dataset.SeriesDescription == sd
                                assert df.dataset.SeriesInstanceUID == series_uid

                            target_accession_path = os.path.join(target_root_path, an)
                            print 'Info: series {}: {}'.format(series_path, sd),
                            if is_t2w:
                                print 'is t2w'
                                clean_t2w(df_list, target_accession_path)
                                t2w_counter += 1
                            elif is_t2wfs:
                                print 'is t2wfs'
                                clean_t2wfs(df_list, target_accession_path)
                                t2wfs_counter += 1
                            elif is_dwi:
                                print 'is dwi'
                                clean_dwi(df_list, target_accession_path)
                                dwi_counter += 1
                            elif is_dce:
                                print 'is dce'
                                clean_dce(df_list, target_accession_path)
                                dce_counter += 1
                            else:
                                raise Exception('not is_other but not used')
                            cleaned_series_uid_dict[series_uid] = df_list[0].file_path
                        except:
                            trace_str = traceback.format_exc()
                            print 'Error:', trace_str
                    else:
                        pass
                        # print 'is other'
    print 'Info: All work done'
    print 'Result: t2w_counter: {}'.format(t2w_counter)
    print 'Result: t2wfs_counter: {}'.format(t2wfs_counter)
    print 'Result: dwi_counter: {}'.format(dwi_counter)
    print 'Result: dce_counter: {}'.format(dce_counter)
    print_results()

if __name__ == '__main__':
    main()
