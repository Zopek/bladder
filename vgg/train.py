#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 

import config as cfg 
import vgg16

class Train(Object):
	def __init__(self, network, data):
		self.data = data
		self.vgg16 = network
		self.batch_size = cfg.BATCH_SIZE
		self.total_ephoch = cfg.TOTAL_EPHOCH 
		self.train_iteration = int(self.data.size('train') / self.batch_size)
		self.test_iteration = int(self.data.size('test') / self.batch_size)
		self.summary_iter = self.train_iteration / 2
		#self.saver_iter = cfg.SAVER_ITER 
		self.initial_learning_rate = cfg.INITIAL_LEARNING_RATE

		#self.variable_to_restore = tf.global_variables()
		#self.saver = tf.train.Saver(sel.variable_to_restore)

		self.global_step = tf.Variable(0, trainable = False)
		self.learning_rate = tf.train.piecewise_constant(self.global_step, [self.total_ephoch], [self.initial_learning_rate])
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.vgg16.cost, global_step = self.global_step)

		#self.saver.restore(self.sess, ?)

	def train(self):
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)

			for epoch in self.total_ephoch:

				self.data.shuffle()
				for iteration in self.train_iteration + 1:

					images, labels = self.data.next_batch('train')
					feed_dict_train = {self.vgg16.images: images, self.vgg16.labels: labels}
					_, train_loss = sess.run([self.optimizer, self.vgg16.cost], feed_dict = feed_dict_train)

					if iteration % self.summary_iter == 0:
						train_accuracy = sess.run(self.vgg16.accuracy, feed_dict = feed_dict_train)
						print('Iter:', iteration, 'Loss:', train_loss, 'Train Accuracy:', train_accuracy)

				for iteration in self.test_iteration:

					images, labels = self.data.next_batch('test')
					feed_dict_test = {self.vgg16.images: images, self.vgg16.labels: labels}

					if iteration == 0:
						test_accuracy = sess.run(self.vgg16.accuracy, feed_dict = feed_dict_test)
					else:
						test_accuracy += sess.run(self.vgg16.accuracy, feed_dict = feed_dict_test)

					if iteration == self.test_iteration - 1:
						print('Epoch:', epoch, 'Test Accuracy:', test_accuracy/self.test_iteration)




			