import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import csv
import json
import scipy.ndimage
import pickle

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import config
import utils
import model as modellib
#from mrcnn import visualize
#from mrcnn.model import log

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
'''
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
'''

class BladderConfig(config.Config):
    """Configuration for training on the bladder dataset.
    Derives from the base Config class and overrides values specific
    to the toy bladder dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bladder"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 2 cancer

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 160
    IMAGE_MAX_DIM = 160

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1536 / IMAGES_PER_GPU

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 332 / IMAGES_PER_GPU

config = BladderConfig()

class InferenceConfig(BladderConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

class BladderDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_bladder(self, path):
        """load images.
        path: the path of train or val csv.
        """
        # Add classes
        self.add_class("bladder", 1, "cancer")

        # Add images
        images = []
        size = 0

        with open(path, 'r') as f:
            reader = csv.reader(f)
            for i in reader:

                images.append(i)
                size += 1

        for i in range(size):
            self.add_image("bladder", image_id=i, path=images[i])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        height = 160
        width = 160
        info = self.image_info[image_id]
        data_path = info['path']

        image = np.load(os.path.join('/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order', data_path[0]))
        bladder_bbox = self.read_bladder_bbox(os.path.join('/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order', data_path[2]))

        image_ADC = self.process_one_channel(image[0], bladder_bbox, height, width)
        image_b0 = self.process_one_channel(image[1], bladder_bbox, height, width)
        image_b1000 = self.process_one_channel(image[2], bladder_bbox, height, width)

        processed_image = np.stack([image_ADC, image_b0, image_b1000], axis=2)
        # processed_image.shape = [height, width, 3]

        return processed_image

    def lists2slices(self, tuple_list):
        return [slice(*t) for t in tuple_list]

    def read_bladder_bbox(self, json_file):
        with open(json_file, 'rb') as fd:
            bbox = self.lists2slices(json.load(fd))
        bbox = bbox[0:2]
        return bbox

    def resize(self, image, new_shape):
        resize_factor = []
        for i, s in enumerate(new_shape):

            if s is None:
                resize_factor.append(1)
            else:
                resize_factor.append((s + 1e-3) / image.shape[i])
        # resize_factor = (np.round(new_shape).astype(np.float) + 1e-3) / image.shape
        # +1e-3 to suppress warning of scipy.ndimage.zoom
        new_image = scipy.ndimage.zoom(image, resize_factor, order=1)
        return new_image

    def process_one_channel(self, image, bladder_bbox, height, width):
        mean = np.mean(image)
        std = max(np.std(image), 1e-9)
        new_image = image[bladder_bbox]
        new_image = (new_image - mean) / std
        new_image = self.resize(new_image, (height, width))
        return new_image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        height = 160
        width = 160

        info = self.image_info[image_id]
        data_path = info['path']
        image = np.load(os.path.join('/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order', data_path[0]))
        bladder_bbox = self.read_bladder_bbox(os.path.join('/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order', data_path[2]))
        label = data_path[3]
        cancer_bbox_path = os.path.join('/DATA/data/yjgu/bladder/bladder_labels', data_path[4])

        cancer_bbox = self.read_cancer_bbox(cancer_bbox_path, image.shape[1], image.shape[2], label)
        cancer_bbox = cancer_bbox[bladder_bbox]
        cancer_bbox = self.resize(cancer_bbox, (height, width))
        cancer_bbox = np.expand_dims(cancer_bbox, 2)

        class_ids = np.array([int(label)])

        return cancer_bbox.astype(np.bool), class_ids.astype(np.int32)

    def read_cancer_bbox(self, filename, image_height, image_width, label):
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


#====================================Load Data========================================
train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/mask_rcnn/train.csv'
val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/mask_rcnn/test.csv'

# Training dataset
dataset_train = BladderDataset()
dataset_train.load_bladder(train_path)
dataset_train.prepare()

# Validation dataset
dataset_val = BladderDataset()
dataset_val.load_bladder(val_path)
dataset_val.prepare()

'''
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names
'''

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

#==============================Init=================================================
# Which weights to start with?# Which 
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

#================================Train======================================
# Train the head branches# Train 
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=3, 
            layers='heads',
            augmentation=True)


# Fine tune all layers# Fine  
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all",
            augmentation=True)

# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

#============================Evaluation==================================
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 70)
APs = []
IoUs = []
i = 0
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlap =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    if i == 0:
        print(gt_mask.shape)
        i += 1
        print(overlap.shape)
    IoU = 0
    for j in range(overlap.shape[0]):

        if IoU < overlap[j, 0]:
            IoU = overlap[j, 0]

    APs.append(AP)
    IoUs.append(IoU)
    
print("mAP: ", np.mean(APs))
print("IoU: ", np.mean(IoUs))
