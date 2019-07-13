import numpy as np
import cPickle
import os
import json
import csv

label_path = '/DATA/data/yjgu/bladder/bladder_labels/'
data_path = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order/'

labeled_access_num = os.listdir(label_path)
all_access_num = os.listdir(data_path)

slices = []
for access_num in all_access_num:
    if access_num in labeled_access_num:
        for series in os.listdir(os.path.join(data_path, access_num)):
            json_file = json.load(open(os.path.join(data_path, access_num, series, 'dilated_mask_bbox.json'), 'r'))
            start_slice = np.array(json_file)[2, 0]
            end_slice = np.array(json_file)[2, 1]

            try:
                stacks = os.listdir(os.path.join(label_path, access_num, series))
                labels = cPickle.load(open(os.path.join(label_path, access_num, series, stacks[0], 'label.txt'), 'r'))
                box_label_path = os.path.join(access_num, series, stacks[0])
            except:
                # try to find other series of the same accession
                print(os.path.join(access_num, series))
                labeled_series = []
                for labeled_ser in os.listdir(os.path.join(label_path, access_num)):
                    if series.split('_')[0] in labeled_ser:  # 'dwi_ax_0'
                        labeled_series.append(labeled_ser)
                if len(labeled_series) == 0:
                    print('This series is not labeled!')
                    continue
                elif len(labeled_series) == 1:
                    stacks = os.listdir(os.path.join(label_path, access_num, labeled_series[0]))
                    labels = cPickle.load(open(os.path.join(label_path, access_num, labeled_series[0], stacks[0], 'label.txt'), 'r'))
                    box_label_path = os.path.join(access_num, labeled_series[0], stacks[0])
            for i in range(start_slice, end_slice):
                slices.append((os.path.join(access_num, series, 'image_' + str(i) + '.npy'),
                               os.path.join(access_num, series, 'dilated_mask_' + str(i) + '.npy'),
                               os.path.join(access_num, series, 'dilated_mask_bbox.json'), labels[i],
                               os.path.join(box_label_path, 'box_label_'+str(i)+'.txt')))

    else:
        print('{} is not labeled!'.format(access_num))

print(len(slices))

f = open('/DATA/data/yjgu/bladder/dwi_ax_detection_dataset/all.csv', 'wb')
writer = csv.writer(f)
for i in range(len(slices)):
    writer.writerow(slices[i])
