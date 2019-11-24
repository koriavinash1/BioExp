#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
from .helpers.utils import load_vol_brats
from mpl_toolkits.axes_grid1 import make_axes_locatable

class gram(object):
    def __init__(self, paths, data_loader, 
			slice_range = (0, 0), 
			step = 1, 
			pad = None, 
			nimages = None
			savepath = None):
        self.paths = paths
        self.data_loader = data_loader
        self.pad = pad
        self.slice_range = slice_range
        self.step = step
        self.nimages = nimages
        self.savepath = savepath

    def cal_gram(self, test_image):
        channels = test_image.shape[-1]
        test_vec = test_image.reshape(-1, channels)
        test_vec = (test_vec - test_vec.min())/(test_vec.max() - test_vec.min())

        return np.matmul(test_vec.T, test_vec)/(1.*len(test_vec))

    def get(self):
        test_image, gt = self.data_loader(test_path[0], slicen = 78, padn=self.pad)
        image_shape = test_image.shape
        channels = image_shape[-1]

        gram_template = np.zeros((channels, channels), dtype='float16')
        counter = 0
        for ii in tqdm(range(num_images)):
            for slice_ in range(self.slice_range[0], self.slice_range[1], self.step):
                try:
                    test_image, gt = self.data_loader(test_path[0], slicen = slice_, padn=self.pad)
                    gram_template += get_gram(test_image.astype('float16'))
                    counter += 1
                except : pass

        gram_template = gram_template/ (counter *1.0)
        if self.save_path:
            np.save(self.save_path, gram_template)
        return gram_template
