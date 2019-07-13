'''
import csv
import json
import os

file_path = '/DATA/data/yjgu/bladder/dwi_ax_detection_dataset'
for file in os.listdir(file_path):
	print file
	csv_path = os.path.join(file_path, file)
	csv_file1 = csv.reader(open(csv_path, 'r'))
	for i in csv_file1:
		print i
'''
'''
import json

path = '/DATA/data/yxzhang/WorkSpace/Renji/Data/'
with open(path+'label_data.json','r') as f:
    data = json.load(f)

for key,value in data.items():
	print key
	print value
'''
'''
import numpy as np 

a = [1, 2, 3]
b = np.array(a)

print b
'''
'''
import csv
import os

source_file = '/DATA/data/yjgu/bladder/dwi_ax_detection_dataset/all.csv'

with open(source_file, 'rb') as fd:
	reader = csv.reader(fd)
	slices = list(reader)
	accession_slice_mapping = dict()
count = 0
for s in slices:
	accession = s[0].split('/', 1)[0]
	if count == 0:
		print s
		print s[0]
    	count = 1
'''
'''
import json

with open('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/from_senior/bladder_dwi_2d_model/cfgs/test.json','r') as f:
    cfg = json.load(f)

    print "dataset_csv_dir =", cfg['dataset_csv_dir']
    print "image_root_dir =", cfg['image_root_dir']
    print "cancer_bboxes_root_dir =", cfg['cancer_bboxes_root_dir']

    # output dirs
    print "model_weights_dir =", cfg['model_weights_dir']
    print "log_dir =", cfg['log_dir']

    # dataset settings
    print "num_dataloader_workers =", cfg['num_dataloader_workers']
    print "new_height =", cfg['new_height']
    print "new_width =", cfg['new_width']
    print "using_bladder_mask =", cfg['using_bladder_mask']
    print "caching_data =", cfg['caching_data']
    print "batch_size =", cfg['batch_size']

    # model settings
    print "mil_pooling_type =", cfg['mil_pooling_type']
    print "concat_pred_list =", cfg['concat_pred_list']
    print "num_shared_encoders =", cfg['num_shared_encoders']

    # training configurations
    print "num_step_one_epoches =", cfg['num_step_one_epoches']
    print "num_step_two_epoches =", cfg['num_step_two_epoches']
    print "base_lr =", cfg['base_lr']
    print "loss_weights_list =", cfg['loss_weights_list']
    print "dropout_prob_list =", cfg['dropout_prob_list']
    print "weight_decay =", cfg['weight_decay']
'''
'''
import csv
import numpy as np 

csv_file = '/DATA/data/yjgu/bladder/dwi_ax_detection_dataset/0_cv_val.csv'

def read_csv(csv_file):
    with open(csv_file, 'rb') as fd:
        reader = csv.reader(fd)
        return np.array(list(reader))

samples = read_csv(csv_file)
label = samples[:, 3].astype(np.int)
samples = samples[label != 2]

image = samples[:, 0]
print type(image)
print image.shape
print image[0]
'''

import json
import numpy as np 

def lists2slices(tuple_list):
    return [slice(*t) for t in tuple_list]

json_file = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order/D0564603/dwi_ax_0/dilated_mask_bbox.json'
with open(json_file, 'rb') as fd:
	a = json.load(fd)
	print a
	bbox = lists2slices(a)[0:2]
	print type(bbox[0])

b = np.load('/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order/D0564603/dwi_ax_0/dilated_mask_12.npy')
b = b[bbox]
print b.shape
