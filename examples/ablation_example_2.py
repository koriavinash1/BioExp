import sys
sys.path.append('..')
import argparse
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
# from tqdm import tqdm
from glob import glob
from BioExp.helpers import utils
from BioExp.spatial import ablation
import os

from keras.backend.tensorflow_backend import set_session
from BioExp.helpers.metrics import *
from BioExp.helpers.losses import *

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

	
parser = argparse.ArgumentParser(description='feature study')
parser.add_argument('--seq', default='flair', type=str, help='mri sequence')
parser = parser.parse_args()


model_path        = '/media/balaji/CamelyonProject/parth/checkpoints/double_headed_autoencoder/autoencoder_double_headed_no_skip.hdf5'
# weights_path      = '../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)

results_root_path = './results/'
data_root_path = '/media/balaji/CamelyonProject/parth/brats_2018/val'


layers_to_consider = ['conv2d_2']
input_name = 'input_1'

#########################################################################

feature_maps = []
layers = []
classes = []


infoclasses = {}
for i in range(4): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3,)
infoclasses['ET'] = (3,)
infoclasses['CT'] = (1,3,)
num_images = 2
metric = dice_label_coef ## (gt, pred, class_)

model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
	                                'dice_whole_metric':dice_whole_metric,
	                                'dice_core_metric':dice_core_metric,
	                                'dice_en_metric':dice_en_metric})

weights_path = '/media/balaji/CamelyonProject/parth/checkpoints/double_headed_autoencoder/autoencoder_double_headed_no_skip_weights.hdf5'
model.save_weights(weights_path)

for layer_name in layers_to_consider:	
	# for i, file in enumerate(glob(data_root_path +'*')[5:5	+num_images]):

	model.load_weights(weights_path)
	image, gt = utils.load_vol_brats('/media/balaji/CamelyonProject/parth/brats_2018/val/Brats18_2013_3_1', slicen=78)

	A = ablation.Ablate(model, weights_path, metric, layer_name, image[None,...], gt, classes = infoclasses, image_name=str(i))
	path = 'results/'
	os.makedirs(path, exist_ok=True)


	df1 = A.ablate_filters(save_path='./results/', step=2)
	# 	if i == 0: 
	# 		df = df1
	# 	else: 
	# 		df.iloc[:,1:] += df1.iloc[:,1:]
	
	# df.iloc[:,1:] = df.iloc[:,1:]/(1. * num_images)
	print (df1)
