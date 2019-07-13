import pydicom
import os
import numpy as np

class DicomFile:
    def __init__(self, file_path, dataset):
        self.file_path = file_path
        self.dataset = dataset

filepath = '/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder/2013-2015/D0501566/HKSY1EDT/0CLM3XFI/I1000000'
df = DicomFile(filepath, pydicom.read_file(filepath))
print df.dataset.AccessionNumber
print df.dataset.SeriesDescription
print df.dataset.SeriesInstanceUID
print df.dataset.ImagePositionPatient
print df.dataset.InstanceNumber
print df.dataset.ImageOrientationPatient
print df.dataset.Rows
print df.dataset.Columns
print df.dataset.SliceLocation
print df.dataset.PixelSpacing
print df.dataset.SpacingBetweenSlices
print df.dataset.pixel_array.T.shape

'''
type(df.dataset.PixelSpacing) = <class 'pydicom.multival.MultiValue'>
'''
'''
a = []
a.append(df.dataset.PixelSpacing[0])
a.append(df.dataset.PixelSpacing[1])
spacing = np.array([df.dataset.SpacingBetweenSlices] + a)

b = np.array([1,1,1])
print spacing.shape, b.shape
'''