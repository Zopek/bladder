from __future__ import print_function
import json
import math
import os
import dicom
import glob
import pickle
import csv
import numpy as np
import traceback

# This file is for extraction of labels and bounding boxes in each slice.
# The labels are stored as label.txt which is a list and each element is the corresponding label for each slice.
# The boxes are stored as box_label.txt which is also a list [[box for slice1],[box for slice2],[box for slice3], ......]
# [box for slice] = [[x, y, r], [x, y, r]...] and x, y, r are all pixel coordinate.
# The length of [box for slice] represents the number of tumors in this slice and if there is no tumor in the slice, it will be [].

classes_path = '/DATA3_DB7/data/public/renji_data/labels/bladder_tags_class.csv'
reader1 = csv.reader(open(classes_path, 'r'))
classes = {}
for line in reader1:
    classes[line[0]] = line[1]

path = '/DATA/data/yxzhang/WorkSpace/Renji/Data/'
with open(path + 'label_data.json', 'r') as f:
    label_data = json.load(f)

accession_json = []
record_nodule = {}
for key, value in label_data.items():
    accession_json.append(value['accessionNumber'])
    if len(value["nodules"]) > 0:
        record_nodule[key] = []
        nodules = value['nodules']
        for nodule in nodules:
            try:
                center_loc = nodule['loc']
                r = nodule['r']
                x = nodule['x']
                y = nodule['y']
                record_nodule[str(key)].append([center_loc, r, x, y])
            except:
                print('No loc find for ', value['accessionNumber'])
                print(traceback.print_exc())

print(len(set(accession_json)))  # 1918

savepath = '/DATA/data/yjgu/bladder/bladder_labels/'
datapath = '/DATA3_DB7/data/public/renji_data/bladder_cleaned_distinct_series/'
count1 = 0
count2 = 0
accession_numbers = os.listdir(datapath)
print(len(set(accession_json)))

for accession_number in accession_numbers:
    if accession_number in accession_json and classes[accession_number] == '2':  # cancer sample
        count1 += 1
        accession_path = os.path.join(datapath, accession_number)
        for series_dir in os.listdir(accession_path):
            series_path = os.path.join(accession_path, series_dir)
            for stack_dir in os.listdir(series_path):
                stack_path = os.path.join(accession_path, series_dir, stack_dir)
                files = glob.glob(stack_path + '/*.dcm')
                labels = [0] * len(files)
                slice_locations = []

                ds = dicom.read_file(files[0])
                series_uid = str(ds.SeriesInstanceUID)

                if series_uid in record_nodule.keys():

                    accession_number_json = label_data[series_uid]['accessionNumber']
                    assert accession_number_json == accession_number
                    nodules = record_nodule[series_uid]
                else:
                    print(accession_number, series_dir, stack_dir)
                    continue

                save_label_path = os.path.join(savepath, accession_number, series_dir, stack_dir)
                if not os.path.exists(save_label_path):
                    os.makedirs(save_label_path)

                for i in range(len(files)):
                    file_name = files[i]
                    ds = dicom.read_file(file_name)
                    slice_location = ds.SliceLocation
                    slice_locations.append(slice_location)
                order = np.argsort(np.array(slice_locations))
                for i in range(len(order)):
                    index = order[i]
                    file_name = files[index]
                    ds = dicom.read_file(file_name)
                    slice_location = ds.SliceLocation
                    pixel_space = ds.PixelSpacing[0]
                    boxes_on_this_slice = []

                    for nodule in nodules:
                        [center_loc, r, x, y] = nodule
                        if center_loc - r / 2.0 < slice_location < center_loc + r / 2.0:  # the slice is inner positive
                            labels[i] = 1
                            box_r = math.sqrt(r ** 2 - (slice_location - center_loc) ** 2) / pixel_space
                            boxes_on_this_slice.append([x, y, box_r])
                        elif center_loc - r < slice_location < center_loc + r:  # the slice is outer positive
                            if labels[i] != 1:
                                labels[i] = 2
                            box_r = math.sqrt(r ** 2 - (slice_location - center_loc) ** 2) / pixel_space
                            boxes_on_this_slice.append([x, y, box_r])

                    pickle.dump(boxes_on_this_slice, open(os.path.join(save_label_path, 'box_label_' + str(i) + '.txt'), 'wa'))

                pickle.dump(labels, open(os.path.join(save_label_path, 'label.txt'), 'wa'))

    elif classes[accession_number] == '0' or classes[accession_number] == '1':  # normal sample
        count2 += 1
        accession_path = os.path.join(datapath, accession_number)
        for series_dir in os.listdir(accession_path):
            series_path = os.path.join(accession_path, series_dir)
            for stack_dir in os.listdir(series_path):
                stack_path = os.path.join(accession_path, series_dir, stack_dir)
                files = glob.glob(stack_path + '/*.dcm')
                labels = [0] * len(files)
                save_label_path = os.path.join(savepath, accession_number, series_dir, stack_dir)
                if not os.path.exists(save_label_path):
                    os.makedirs(save_label_path)
                pickle.dump(labels, open(os.path.join(save_label_path, 'label.txt'), 'wa'))
                for i in range(len(files)):
                    pickle.dump([], open(os.path.join(save_label_path, 'box_label_' + str(i) + '.txt'), 'wa'))

print(count1)
print(count2)
print(count1 + count2)
