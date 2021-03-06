#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

import train
import vgg16
import preprocess

def main():
	network = vgg16.vgg16()
	data_loader = preprocess.data_preprocess()

	train_network = train.Train(network, data_loader)
	train_network.train()

if __name__ == '__main__':
	main()

