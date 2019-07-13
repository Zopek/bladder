import numpy as np
import os
import pydicom as dicom


dwi_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/dwi_ax_0/stack0_b0guess/I_0.dcm"
dwi1_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/dwi_ax_0/stack0_b0guess/I_18.dcm"
t2w_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/t2w_ax_0/stack0/I_0.dcm"
t2w1_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/t2w_ax_0/stack0/I_18.dcm"
t2w2_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/t2w_ax_0/stack0/I_19.dcm"
t2wfs_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/t2wfs_ax_0/stack0/I_0.dcm"
t2wfs1_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/t2wfs_ax_0/stack0/I_18.dcm"
t2wfs2_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/D1241536/t2wfs_ax_0/stack0/I_19.dcm"

dwi = dicom.read_file(dwi_path)
dwi1 = dicom.read_file(dwi1_path)
t2w = dicom.read_file(t2w_path)
t2w1 = dicom.read_file(t2w1_path)
t2w2 = dicom.read_file(t2w2_path)
t2wfs = dicom.read_file(t2wfs_path)
t2wfs1 = dicom.read_file(t2wfs1_path)
t2wfs2 = dicom.read_file(t2wfs2_path)

print("dwi", dwi.ImagePositionPatient)
print("t2w", t2w.ImagePositionPatient)
print("t2wfs", t2wfs.ImagePositionPatient)

print("dwi1", dwi.SliceLocation)
print("dwi2", dwi1.SliceLocation)
print("t2w1", t2w.SliceLocation)
print("t2w2", t2w1.SliceLocation)
print("t2w3", t2w2.SliceLocation)
print("t2wfs1", t2wfs.SliceLocation)
print("t2wfs2", t2wfs1.SliceLocation)
print("t2wfs3", t2wfs2.SliceLocation)
'''
path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series/9007414997"

mode_dict = {}

for mode in os.listdir(path):

	if not mode_dict.has_key(mode):
		mode_dict[mode] = None

	mode_path = os.path.join(path, mode)

	stack_path = os.path.join(mode_path, os.listdir(mode_path)[0])
	filename = "I_1.dcm"
	file_path = os.path.join(stack_path, filename)
	image = dicom.read_file(file_path)

	mode_dict[mode] = image

for key, value in mode_dict.items():

	print(key, value.ImagePositionPatient)
'''