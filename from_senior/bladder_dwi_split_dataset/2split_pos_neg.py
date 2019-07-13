import numpy as np
import cPickle
import os
import json
import csv

label_path = '/DATA3_DB7/data/public/renji_data/bladder_cleaned_distinct_series_labels/'
data_path = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_float32/'

labeled_access_num = os.listdir(label_path)
all_access_num = os.listdir(data_path)

positive_slices = []
negative_slices = []
for access_num in all_access_num:
    if access_num in labeled_access_num:
        for series in os.listdir(os.path.join(data_path, access_num)):
            json_file = json.load(open(os.path.join(data_path, access_num, series, 'dilated_mask_bbox.json'), 'r'))
            start_slice = np.array(json_file)[2, 0]
            end_slice = np.array(json_file)[2, 1]

            try:
                stacks = os.listdir(os.path.join(label_path, access_num, series))
                labels = cPickle.load(open(os.path.join(label_path, access_num, series, stacks[0], 'label.txt'), 'r'))
            except:
                print os.path.join(access_num, series)
                labeled_series = []
                for labeled_ser in os.listdir(os.path.join(label_path, access_num)):
                    if series.split('_')[0] in labeled_ser:  # 'dwi_ax_0'
                        labeled_series.append(labeled_ser)
                if len(labeled_series) == 0:
                    print 'This series is not labeled!'
                    continue
                elif len(labeled_series) == 1:
                    stacks = os.listdir(os.path.join(label_path, access_num, labeled_series[0]))
                    labels = cPickle.load(open(os.path.join(label_path, access_num, labeled_series[0], stacks[0], 'label.txt'), 'r'))

            for i in range(start_slice, end_slice):
                if labels[i] == 1:
                    positive_slices.append((os.path.join(access_num, series, 'image_' + str(i) + '.npy'),
                                            os.path.join(access_num, series, 'dilated_mask_' + str(i) + '.npy'),
                                            os.path.join(access_num, series, 'dilated_mask_bbox.json'), 1))
                else:
                    negative_slices.append((os.path.join(access_num, series, 'image_' + str(i) + '.npy'),
                                            os.path.join(access_num, series, 'dilated_mask_' + str(i) + '.npy'),
                                            os.path.join(access_num, series, 'dilated_mask_bbox.json'), 0))

    else:
        print '{} is not labeled!'.format(access_num)
        # for series in os.listdir(os.path.join(data_path, access_num)):
        #     json_file = json.load(open(os.path.join(data_path, access_num, series, 'dilated_mask_bbox.json'), 'r'))
        #     start_slice = np.array(json_file)[2, 0]
        #     end_slice = np.array(json_file)[2, 1]
        #     for i in range(start_slice, end_slice):
        #         negative_slices.append((os.path.join(access_num, series, 'image_' + str(i) + '.npy'),
        #                                 os.path.join(access_num, series, 'dilated_mask_' + str(i) + '.npy'),
        #                                 os.path.join(access_num, series, 'dilated_mask_bbox.json'), 0))

print len(positive_slices)
print len(negative_slices)

# 1921
# 11682

f = open('/DATA3_DB7/data/public/renji_data/splits_accession/positive_slices.csv', 'wb')
writer = csv.writer(f)
for i in range(len(positive_slices)):
    writer.writerow(positive_slices[i])

f = open('/DATA3_DB7/data/public/renji_data/splits_accession/negative_slices.csv', 'wb')
writer = csv.writer(f)
for i in range(len(negative_slices)):
    writer.writerow(negative_slices[i])

cPickle.dump(positive_slices, open('/DATA3_DB7/data/public/renji_data/splits_accession/positive_slices.txt', 'wb'))
cPickle.dump(negative_slices, open('/DATA3_DB7/data/public/renji_data/splits_accession/negative_slices.txt', 'wb'))
