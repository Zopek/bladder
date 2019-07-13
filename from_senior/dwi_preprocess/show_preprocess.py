import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plot_3D_image
import full_preprocess
import preprocess_util
import os
import pickle
from scipy import ndimage
def read_cancer_bbox(filename, image_height, image_width):
    with open(filename, 'r') as f:
        cancer_bboxes = pickle.load(f)

    bboxes_image = np.zeros((image_height, image_width))
    grid_x, grid_y = np.mgrid[0:image_height, 0:image_width]
    for box in cancer_bboxes:

        x = box[0]
        y = box[1]
        r = box[2]
        dist_from_center = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
        mask = dist_from_center < r
        bboxes_image = np.logical_or(bboxes_image, mask)

    return bboxes_image.astype(np.int)

if __name__ == '__main__':
    labels_path = "/DB/rhome/qyzheng/Desktop/Link to renji_data/bladder_cleaned_distinct_series_labels"
    root_dir = '/DATA3_DB7/data/public/renji_data/bladder_cleaned_distinct_series/'
    series_dir_list = []
    for i in range(10):

        accession = os.listdir(root_dir)
        accession_path = os.path.join(root_dir, accession[i])
        if "dwi_ax_0" in os.listdir(accession_path):
            series_dir_list.append(os.path.join(accession[i], "dwi_ax_0"))
    '''
    series_dir_list = [
        # 'D0719976/dwi_ax_0',
        # 'D1624257/dwi_ax_0',
        # 'D2049831/dwi_ax_0',
        # 'D2291117/dwi_ax_0',
        # 'D0645094/dwi_ax_1',
        # 'W0312093/dwi_ax_0',
        # 'D2314620/dwi_ax_0',
        # 'D0667043/dwi_ax_0',
        # 'D1598531/dwi_ax_0',
        # 'D1647256/dwi_ax_0',
        # 'D0501566/dwi_ax_0',
        # '/D0643108/dwi_ax_0',
        '/D1573459/dwi_ax_0'
    ]
    '''
    for series_dir in series_dir_list:
        whole_adc, whole_b0, whole_b1000, mask, dilated_mask \
            = full_preprocess.full_preprocess(root_dir + series_dir, True)
            
        stack = os.listdir(root_dir + series_dir)
        label_path = os.path.join(labels_path, series_dir, stack[0])
        labels = pickle.load(open(os.path.join(label_path, 'label.txt'), 'r'))
        for j in range(len(labels)):

            if labels[j] == 0:
                continue

    for series_dir in series_dir_list:
        whole_adc_1, whole_b0_1, whole_b1000_1, mask_1, dilated_mask_1 \
            = full_preprocess.full_preprocess(root_dir + series_dir, True)

        stack = os.listdir(root_dir + series_dir)
        label_path = os.path.join(labels_path, series_dir, stack[0])
        labels = pickle.load(open(os.path.join(label_path, 'label.txt'), 'r'))
        for j in range(len(labels)):

            if labels[j] == 0:
                continue

            #print(type(whole_adc[0]))
            whole_adc = whole_adc_1[:,:,j][:,:,np.newaxis]
            whole_b0 = whole_b0_1[:,:,j][:,:,np.newaxis]
            whole_b1000 = whole_b1000_1[:,:,j][:,:,np.newaxis]
            mask = mask_1[:,:,j][:,:,np.newaxis]
            dilated_mask = dilated_mask_1[:,:,j][:,:,np.newaxis]

            masked_adc = whole_adc * mask
            masked_b0 = whole_b0 * mask
            dilated_masked_adc = whole_adc * dilated_mask
            dilated_masked_b0 = whole_b0 * dilated_mask
            mask = mask[preprocess_util.find_bounding_box(mask)]


            def transpose_and_flip(a):
                return np.flip(np.flip(a.transpose((2, 1, 0)), 1), 2)


            '''
            a = whole_adc
            print(a.shape)
            cancer_bbox_path = os.path.join('/DATA/data/yjgu/bladder/bladder_labels', 'D1573459/dwi_ax_1/stack2_b1000/box_label_4.txt')
            cancer_bbox = read_cancer_bbox(cancer_bbox_path, a.shape[0], a.shape[1])

            image = whole_adc[:,:,4]
            vmin = np.min(image)
            vmax = np.max(image)

            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            fig.colorbar(ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
            
            image = dilated_masked_adc[:,:,4]
            vmin = np.min(image)
            vmax = np.max(image)
            image[cancer_bbox != 0] = 1

            #objects = list(ndimage.find_objects(image))
            #print(objects)

            ax = fig.add_subplot(2, 2, 2)
            fig.colorbar(ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cm.get_cmap('nipy_spectral'), animated=True), ax=ax)
            
            ax = fig.add_subplot(2, 2, 3)
            ax.imshow(cancer_bbox)
            fig.show()
            plt.show()
            plt.savefig('./sample8.png')
            '''
            '''
            dilated_mask_bbox = preprocess_util.find_bounding_box(dilated_mask)
            def find_bounding_box(a):
                objects = list(ndimage.find_objects(image))
                assert len(objects) == 1
                return objects[0]
            '''

            fig = plt.figure()
            plane = plot_3D_image.Multi3DArrayPlane(fig, 3, 3)
            plane.add(transpose_and_flip(masked_adc))
            plane.add(transpose_and_flip(dilated_masked_adc))
            plane.add(transpose_and_flip(whole_adc))

            plane.add(transpose_and_flip(masked_b0))
            plane.add(transpose_and_flip(dilated_masked_b0))
            plane.add(transpose_and_flip(whole_b0))

            plane.add(transpose_and_flip(mask))
            plane.add(transpose_and_flip(dilated_mask))
            plane.add(transpose_and_flip(whole_b1000))
            plane.ready()
            fig.show()
            plt.show()
            plt.savefig('./sample/dwi/sample_{}_{}.png'.format(series_dir.split('/')[0], j))

        '''
        fig = plt.figure()
        ax = fig.add_subplot(3, 3, 1)
        ax.imshow(masked_adc[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 2)
        ax.imshow(dilated_masked_adc[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 3)
        ax.imshow(whole_adc[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 4)
        ax.imshow(masked_b0[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 5)
        ax.imshow(dilated_masked_b0[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 6)
        ax.imshow(whole_b0[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 7)
        ax.imshow(mask[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 8)
        ax.imshow(dilated_mask[0,:,:], 'gray')
        ax = fig.add_subplot(3, 3, 9)
        ax.imshow(whole_b1000[0,:,:], 'gray')
        plt.show()
        plt.savefig('./sample7.png')
        '''
    #plt.show()
