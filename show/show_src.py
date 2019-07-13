import numpy as np
import os, csv, random, gc, pickle
import nibabel as nib

img_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/Brats17_CBICA_ARF_1_t1ce.nii.gz"
lab_path = "/DB/rhome/qyzheng/Desktop/qyzheng/source/Brats17_CBICA_ARF_1_seg_4c.nii.gz"

img = nib.load(img_path).get_data()
img = img.astype(np.float32)
print img.shape
print img

seg_img = nib.load(lab_path).get_data()
print seg_img.shape
print seg_img.max()