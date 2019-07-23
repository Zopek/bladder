#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 

import config as cfg 

class vgg16(object):
    def __init__(self, isTraining = True):
        self.num_class = cfg.NUM_CLASS 
        self.batch_size = cfg.BATCH_SIZE
        self.height = cfg.HEIGHT 
        self.width = cfg.WIDTH 
        self.channel = cfg.CHANNEL 

        self.images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel], name = 'images')
        self.labels = tf.placeholder(tf.float32, [None, self.num_class], name = 'labels')
        self.logits = self.network(self.images)

        if isTraining:
            self.cost = self.loss(self.logits, self.labels)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def network(self, inputs):
        net = self.conv_layer(inputs, [3, 3, self.channel, 64], name = 'conv1_1')
        net = self.conv_layer(net, [3, 3, 64, 64], name = 'conv1_2')
        net = self.pooling_layer(net, name = 'maxpool1')

        net = self.conv_layer(net, [3, 3, 64, 128], name = 'conv2_1')
        net = self.conv_layer(net, [3, 3, 128, 128], name = 'conv2_2')
        net = self.pooling_layer(net, name = 'maxpool2')

        net = self.conv_layer(net, [3, 3, 128, 256], name = 'conv3_1')
        net = self.conv_layer(net, [3, 3, 256, 256], name = 'conv3_2')
        net = self.conv_layer(net, [3, 3, 256, 256], name = 'conv3_3')
        net = self.pooling_layer(net, name = 'maxpool3')

        net = self.conv_layer(net, [3, 3, 256, 512], name = 'conv4_1')
        net = self.conv_layer(net, [3, 3, 512, 512], name = 'conv4_2')
        net = self.conv_layer(net, [3, 3, 512, 512], name = 'conv4_3')
        net = self.pooling_layer(net, name = 'maxpool4')

        net = self.conv_layer(net, [3, 3, 512, 512], name = 'conv5_1')
        net = self.conv_layer(net, [3, 3, 512, 512], name = 'conv5_2')
        net = self.conv_layer(net, [3, 3, 512, 512], name = 'conv5_3')
        net = self.pooling_layer(net, name = 'maxpool4')

        net = self.fc_layer(net, 4096, name = 'fc1')
        net = self.fc_layer(net, 4096, name = 'fc2')
        net = self.fc_layer(net, self.num_class, name = 'fc3')

        net = self.softmax(net, name = 'softmax')

        return net

    def conv_layer(self, inputs, shape, BN = True, name = '0'):
        weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = 'weight')
        biases = tf.Variable(tf.constant(0.1, shape = [shape[3]]), name = 'biases')

        conv = tf.nn.conv2d(inputs, weight, strides = [1, 1, 1, 1], padding = 'SAME', name = name)

        if BN:
            depth = shape[3]
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

            conv_bn = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
            conv = tf.add(conv_bn, biases)
            conv = tf.nn.relu(conv)
        else:
            conv = tf.add(conv, biases)
            conv = tf.nn.relu(conv)

        return conv

    def pooling_layer(self, inputs, name = '0'):
        pool = tf.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

        return pool

    def fc_layer(self, inputs, output_channel, BN = True, name = '0'):
        shape = int(np.prod(inputs.get_shape()[1:]))
        weight = tf.Variable(tf.truncated_normal([shape, output_channel], stddev = 0.1), name = 'weight')
        biases = tf.Variable(tf.constant(0.1, shape = [output_channel]), name = 'biases')

        inputs_flat = tf.reshape(inputs, [-1, shape])
        fc = tf.matmul(inputs_flat, weight) + biases

        if BN:
            depth = output_channel
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

            fc_bn = tf.nn.batch_normalization(fc, mean, variance, shift, scale, 1e-05)
            fc = tf.nn.relu(fc_bn, name = name)
        else:
            fc = tf.nn.relu(fc, name = name)

        return fc

    def softmax(self, inputs, name = '0'):

        return tf.nn.softmax(inputs, name = name)

    def loss(self, logits, labels):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

        return cost