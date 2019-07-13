#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf 
import tensorlayer as tl
import numpy as np
import os
import csv
import random
import json
from scipy import ndimage
import time
import pickle

import model
import augmentation

data_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
label_path = '/DATA/data/yjgu/bladder/bladder_labels/'

def lists2slices(tuple_list):
	return [slice(*t) for t in tuple_list]

def read_bladder_bbox(json_file):
	with open(json_file, 'rb') as fd:
		bbox = lists2slices(json.load(fd))
	bbox = bbox[0:2]
	return bbox

def resize(image, new_shape):
	resize_factor = []
	for i, s in enumerate(new_shape):

		if s is None:
			resize_factor.append(1)
		else:
			resize_factor.append((s + 1e-3) / image.shape[i])
	# resize_factor = (np.round(new_shape).astype(np.float) + 1e-3) / image.shape
	# +1e-3 to suppress warning of scipy.ndimage.zoom
	new_image = ndimage.zoom(image, resize_factor, order=1)
	return new_image

def process_one_channel(image, bladder_bbox, height, width):
	mean = np.mean(image)
	std = max(np.std(image), 1e-9)
	new_image = image[bladder_bbox]
	new_image = (new_image - mean) / std
	new_image = resize(new_image, (height, width))
	return new_image

def read_cancer_bbox(root_dir, filename, sizes):

    dwi_size = int(sizes[0])
    for i in range(len(filename)):

        if filename[i].split('/')[1] == 'dwi_ax_0':
            origin_size = int(sizes[0])
        elif filename[i].split('/')[1] == 't2w_ax_0':
            origin_size = int(sizes[1])
        else:
            assert filename[i].split('/')[1] == 't2wfs_ax_0'
            origin_size = int(sizes[2])

        file_path = os.path.join(root_dir, filename[i])
        with open(file_path, 'r') as f:
            cancer_bboxes = pickle.load(f)

        grid_x, grid_y = np.mgrid[0:origin_size, 0:origin_size]
        bboxes_image = np.zeros((origin_size, origin_size))
        for box in cancer_bboxes:
            
            x = box[0]
            y = box[1]
            r = box[2]
            dist_from_center = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
            mask = dist_from_center < r
            bboxes_image = np.logical_or(bboxes_image, mask)

        if filename[i].split('/')[1] != 'dwi_ax_0':
            resize_factor = (np.array((dwi_size, dwi_size)).astype(np.float) + 1e-3) / bboxes_image.shape  # +1e-3 to suppress warning of scipy.ndimage.zoom
            bboxes_image = ndimage.interpolation.zoom(bboxes_image.astype(np.float), resize_factor, order=0)

        if i == 0:
            bboxes = bboxes_image
        else:
            bboxes = np.logical_or(bboxes, bboxes_image)
    
    return bboxes.astype(np.int)

def shuffle(dataset):
	random.shuffle(dataset)
	return dataset

def next_batch(dataset, batch_size, height, width, epoch, mode):
	images = []
	labels = []
	accessions = []
	for i in range(batch_size):

		# dataset[0] = ['D1867766/dwi_ax_0/image_3.npy', 
		#				'D1867766/dwi_ax_0/dilated_mask_3.npy',
		# 				'D1867766/dwi_ax_0/dilated_mask_bbox.json', '0', 
		#				'D1867766/dwi_ax_0/stack0_b0guess/box_label_3.txt']
		ind = batch_size * epoch + i
		image = np.load(os.path.join(data_path, dataset[ind][0]))
		bladder_bbox = read_bladder_bbox(os.path.join(data_path, dataset[ind][1]))
		# label = dataset[ind][2]
		cancer_bbox_path = dataset[ind][3:6]
		sizes = dataset[ind][6:]
		accession = dataset[ind][0].split('/')[0] + '_' + dataset[ind][0].split('/')[1].split('.')[0]

		image_ADC = process_one_channel(image[0], bladder_bbox, height, width)
		image_b0 = process_one_channel(image[1], bladder_bbox, height, width)
		image_b1000 = process_one_channel(image[2], bladder_bbox, height, width)
		image_t2w = process_one_channel(image[3], bladder_bbox, height, width)
		image_t2wfs = process_one_channel(image[4], bladder_bbox, height, width)

		processed_image = np.stack([image_ADC, image_b0, image_b1000, image_t2w, image_t2wfs], axis=2)
		# processed_image.shape = [height, width, 3]

		cancer_bbox = read_cancer_bbox(label_path, cancer_bbox_path, sizes)
		cancer_bbox = cancer_bbox[bladder_bbox]
		cancer_bbox = resize(cancer_bbox, (height, width))
		cancer_bbox = np.expand_dims(cancer_bbox, 2)
		# cancer_bbox.shape = [height, width, 1]

		if mode == 'train':
			aug_image, aug_label = augmentation.random_transform(processed_image, cancer_bbox)
		elif mode == 'val':
			aug_image = processed_image
			aug_label = cancer_bbox

		images.append(aug_image)
		labels.append(aug_label)
        accessions.append(accession)

	images = np.asarray(images, dtype=np.float32)
	labels = np.asarray(labels)

	return images, labels, accessions

def vis_imgs2(X, y_, y, path, i):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], X[:,:,4,np.newaxis]]), size=(1, 5),
        image_path=os.path.join(path, 'image_{}.png'.format(i)))
    tl.vis.save_images(np.asarray([y_, y]), size=(1, 2), image_path=os.path.join(path, 'label_{}.png'.format(i)))

def main():

    ## Create folder to save trained model and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples")

    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 10
    lr = 0.0001 
    # lr_decay = 0.5
    # decay_every = 10
    beta1 = 0.9
    n_epoch = 180
    print_freq_step = 100
    height = 160
    width = 160

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/0_label_1_train_val.csv'
    val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/unet_brain/label_1_test.csv'
    train = []
    val = []
    train_size = 0
    val_size = 0
    
    with open(train_path, 'r') as f:
    	reader = csv.reader(f)
    	for i in reader:

    		train.append(i)
    		train_size += 1

    with open(val_path, 'r') as f:
    	reader = csv.reader(f)
    	for i in reader:

    		val.append(i)
    		val_size += 1

    train_epoch = train_size / batch_size
    val_epoch = val_size / batch_size

    ###======================== SHOW DATA ===================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ###======================== DEFIINE MODEL =======================###
    ## nz is 5 as we input ADC, b0, b1000, t2w, t2wfs
    t_image = tf.placeholder('float32', [batch_size, height, width, 5], name='input_image')
    ## labels are either 0 or 1
    t_seg = tf.placeholder('float32', [batch_size, height, width, 1], name='target_segment')
    ## train inference
    net = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
    ## test inference
    net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)

    ###======================== DEFINE LOSS =========================###
    ## train losses
    out_seg = net.outputs
    dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
    iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
    dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
    loss = dice_loss

    ## test losses
    test_out_seg = net_test.outputs
    test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
    test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
    test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])

    ###======================== DEFINE TRAIN OPTS =======================###
    t_vars = tl.layers.get_variables_with_name('u_net', True, True)
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)

    ###======================== LOAD MODEL ==============================###
    tl.layers.initialize_global_variables(sess)
    ## load existing model if possible
    # tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net.npz', network=net)

    ###======================== TRAINING ================================###
    iou = 0
    for epoch in range(0, n_epoch+1):

        epoch_time = time.time()
        
        ## update decay learning rate at the beginning of a epoch
        '''
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr))
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)
		'''

        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        shuffle(train)
        shuffle(val)

        for i in range(train_epoch):

            images, labels, _ = next_batch(train, batch_size, height, width, i, 'val')
            step_time = time.time()

            ## update network
            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    {t_image: images, t_seg: labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

            ## you can show the predition here:
            # vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
            # exit()

            # if _dice == 1: # DEBUG
            #     print("DEBUG")
            #     vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))

            if n_batch % print_freq_step == 0:
                print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
                % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))

            ## check model fail
            if np.isnan(_dice):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")

        print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
                (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

        '''
        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
        '''

        ###======================== VALIDATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for i in range(val_epoch):

            val_images, val_labels, val_accessions = next_batch(val, batch_size, height, width, i, 'val')
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    {t_image: val_images, t_seg: val_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

        print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))

        '''        
        ## save a predition of test set
        if total_iou/n_batch > 0.54:
            if not os.path.exists("samples_paper/{}_{}".format(total_iou/n_batch, epoch)):
                os.makedirs("samples_paper/{}_{}".format(total_iou/n_batch, epoch))
            for i in range(batch_size):

                vis_imgs2(val_images[i], val_labels[i], out[i], "samples_paper/{}_{}".format(total_iou/n_batch, epoch), i)
        '''

        ###======================== SAVE THE BEST MODEL ==========================###
        if iou < total_iou/n_batch:
            iou = total_iou/n_batch
            tl.files.save_npz(net.all_params, name=save_dir+'/u_net.npz', sess=sess)

        print("best", iou)

        
if __name__ == "__main__":

    main()

