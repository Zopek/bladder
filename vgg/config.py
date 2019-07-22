#!/usr/bin/env python
# -*- coding:utf-8 -*-

NUM_CLASS = 7
BATCH_SIZE = 32
HEIGHT = 160
WIDTH = 160
CHANNEL = 5 + 1

TOTAL_EPHOCH = 100
INITIAL_LEARNING_RATE = 1e-4

TRAIN_PATH = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/period/0_period_train.csv'
TEST_PATH = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/period/period_test.csv'
DATA_PATH = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/dwi_t2w_t2wfs_equal'
LABEL_PATH = '/DATA/data/yjgu/bladder/bladder_labels/'