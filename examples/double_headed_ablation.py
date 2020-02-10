import sys
sys.path.append('..')
import argparse
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
from glob import glob
from BioExp.helpers import utils
from BioExp.spatial import ablation
import os

from keras.backend.tensorflow_backend import set_session
from BioExp.helpers.metrics import *
from BioExp.helpers.losses import *

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

model_path        = '../trained_models/double_headed/flair_ae_no_skip.hdf5'
weights_path      = '../trained_models/double_headed/flair_ae_no_skip_weights.hdf5'

results_root_path = './results/'
data_root_path = '/home/brats/parth/test-data/HGG/'


layers_to_consider = ['conv2d_2', 'conv2d_3', 'conv2d_4']
input_name = 'input_1'

infoclasses = {}
for i in range(4): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3,)
infoclasses['ET'] = (3,)
infoclasses['CT'] = (1,3,)
num_images = 2

model = load_model(model_path, compile=False, custom_objects={'gen_dice_loss':gen_dice_loss,
	                                'dice_whole_metric':dice_whole_metric,
	                                'dice_core_metric':dice_core_metric,
	                                'dice_en_metric':dice_en_metric})


for layer_name in layers_to_consider:	
	for i, file in tqdm(enumerate(glob(data_root_path +'*')[5 : 5 + num_images])):

		model.load_weights(weights_path)
		image, gt = utils.load_vol_brats(file, slicen=78)
		image = image[None, ...]

		A = ablation.Ablate(model, weights_path, dice_label_coef, layer_name, image, gt, classes = infoclasses, image_name=str(i))
		path = 'results/'
		os.makedirs(path, exist_ok=True)


		df1 = A.ablate_filters(save_path='./results/', step=4)
		if i == 0: df = df1
		else: df.iloc[:,1:] += df1.iloc[:,1:]
	
	df.iloc[:,1:] = df.iloc[:,1:]/(1. * num_images)
	print (df)
