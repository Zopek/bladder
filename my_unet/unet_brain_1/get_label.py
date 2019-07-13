import numpy as np 
import csv
import cPickle
import os

pos_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/pos_neg/positive_slices.txt'
neg_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/pos_neg/negative_slices.txt'
save_pos_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/pos_label/'
save_neg_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/neg_label/'
data_path = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order/'

pos = cPickle.load(open(pos_path, 'r'))
neg = cPickle.load(open(neg_path, 'r'))

'''
for slice in pos:

	accession_path = slice[0]
	label_path = slice[4]
	accession = accession_path.split('/')[0]

	label = cPickle.load(open(label_path, 'r'))[0]
	x = label[0]
	y = label[1]
	r = int(round(label[2]))

	image_path = os.path.join(data_path, accession_path)
	image = np.load(image_path)
	box = np.zeros((1, image.shape[1], image.shape[2]))
	box[0, (x-r):(x+r), (y-r):(y+r)] = 1
	save_label_path = os.path.join(save_pos_path, accession)
	if not os.path.exists(save_label_path):
		os.makedirs(save_label_path)
	np.save(os.path.join(save_label_path, label_path.split('/')[-1].split('.')[0] + '.npy'), box)
'''

for slice in neg:

	accession_path = slice[0]
	label_path = slice[4]
	accession = accession_path.split('/')[0]

	image_path = os.path.join(data_path, accession_path)
	image = np.load(image_path)
	box = np.zeros((1, image.shape[1], image.shape[2]))
	save_label_path = os.path.join(save_neg_path, accession)
	if not os.path.exists(save_label_path):
		os.makedirs(save_label_path)
	np.save(os.path.join(save_label_path, label_path.split('/')[-1].split('.')[0] + '.npy'), box)