import csv

path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/0_label_1_train.csv'

with open(path, 'rb') as f:
	reader = csv.reader(f)
	for i in reader:
		
		print(i)
		print(type(i))
		print(type(i[0]))
		break