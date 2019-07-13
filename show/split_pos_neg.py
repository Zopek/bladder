import numpy as np
import cPickle
import os
import json
import csv

labels_path = '/DATA/data/yjgu/bladder/bladder_labels'
data_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs_equal.csv'

with open(record_path, 'rb') as f:
    reader = csv.reader(f)

    slices = []
    for accession in reader:

        access_num = accession[0]
        accession_path = os.path.join(data_path, access_num)
        json_file = json.load(open(os.path.join(accession_path, 'dilated_mask_bbox.json'), 'r'))
        start_slice = np.array(json_file)[2, 0]
        end_slice = np.array(json_file)[2, 1]

        label_list = []
        stackname = {"dwi_ax_0": None, "t2w_ax_0": None, "t2wfs_ax_0": None}
        label_path = os.path.join(labels_path, access_num)
        for mode in os.listdir(label_path):

            if mode in stackname.keys():
                mode_path = os.path.join(label_path, mode)
                stack = os.listdir(mode_path)
                stackname[mode] = stack[0]
                stack_path = os.path.join(mode_path, stack[0])
                labels = cPickle.load(open(os.path.join(stack_path, 'label.txt'), 'r'))
                label_list.append(labels)
            else:
                continue
        print(access_num)
        dwi_box_path = os.path.join(access_num, "dwi_ax_0", stackname["dwi_ax_0"])
        t2w_box_path = os.path.join(access_num, "t2w_ax_0", stackname["t2w_ax_0"])
        t2wfs_box_path = os.path.join(access_num, "t2wfs_ax_0", stackname["t2wfs_ax_0"])

        for i in range(start_slice, end_slice):
            slices.append((os.path.join(access_num, 'image_' + str(i) + '.npy'),
                           os.path.join(access_num, 'dilated_mask_bbox.json'), 
                           max(s[i] for s in label_list),
                           os.path.join(dwi_box_path, 'box_label_'+str(i)+'.txt'),
                           os.path.join(t2w_box_path, 'box_label_'+str(i)+'.txt'),
                           os.path.join(t2wfs_box_path, 'box_label_'+str(i)+'.txt'),))

    print(len(slices))

    f = open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/all.csv', 'wb')
    writer = csv.writer(f)
    for i in range(len(slices)):
        writer.writerow(slices[i])

with open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/all.csv', 'rb') as f:
    print len(list(csv.reader(f)))
