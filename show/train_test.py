import csv
import os

record_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/record'
pos_path ='/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/pos_label'
neg_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/neg_label'
data_path = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order'

filenames = os.listdir(record_path)
for filename in filenames:

	if '2' not in filename:
		continue

	f = open('/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/record/{}.csv'.format(filename.split('_')[1]), 'ab')
	writer = csv.writer(f)

	filename = os.path.join(record_path, filename)
	reader = csv.reader(open(filename, 'r'))
	for record in reader:

		img_path = os.path.join(data_path, record[0])
		if 'pos' in filename:
			lab_path = os.path.join(pos_path, record[0].split('/')[0], record[4].split('/')[-1].split('.')[0] + '.npy')
			assert os.path.exists(lab_path)
		else:
			assert 'neg' in filename
			lab_path = os.path.join(neg_path, record[0].split('/')[0], record[4].split('/')[-1].split('.')[0] + '.npy')
			# print lab_path
			assert os.path.exists(lab_path)

		writer.writerow([img_path, lab_path])

	f.close()
