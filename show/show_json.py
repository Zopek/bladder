import json

with open('/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/from_senior/bladder_dwi_2d_model/cfgs/baseline_avg.json', 'rb') as fd:
	cfg = json.load(fd)

for key, value in cfg.items():

	print(key, value)