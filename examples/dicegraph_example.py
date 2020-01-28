import matplotlib
matplotlib.use('Agg')

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
from BioExp.graphs import delta
import os
from pprint import pprint
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from BioExp.helpers.metrics import *
from BioExp.helpers.losses import *

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

	
parser = argparse.ArgumentParser(description='feature study')
parser.add_argument('--seq', default='t1c', type=str, help='mri sequence')
parser = parser.parse_args()


seq_map = {'flair': 0, 't1': 1, 't2': 3, 't1c':2}
seq = parser.seq

print (seq)
model_path        = '../../saved_models/model_{}/model-archi.h5'.format(seq)
weights_path      = '../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)


metric = dice_label_coef
layer_names = [ 'conv2d_1', 'conv2d_3', 'conv2d_5', 'conv2d_7','conv2d_9', 'conv2d_11', 'conv2d_13', 'conv2d_15', 'conv2d_17', 'conv2d_19', 'conv2d_21']


model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
	                                'dice_whole_metric':dice_whole_metric,
	                                'dice_core_metric':dice_core_metric,
	                                'dice_en_metric':dice_en_metric})
model.load_weights(weights_path)


def dataloader(nslice = 78):
	def loader(img_path, mask_path):
		image, gt =  utils.load_vol_brats(img_path, slicen=nslice)
		return image[:,:, seq_map[seq]][:,:, None], gt
	return loader
data_root_path = '/home/brats/parth/test-data/HGG/'

infoclasses = {}
for i in range(4): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3,)
infoclasses['ET'] = (3,)
infoclasses['CT'] = (1,3,)

G = delta.DeltaGraph(model, weights_path, metric, layer_names, classinfo = infoclasses)
json = G.get_concepts('./dicegraph_results')
print (json)

# generate graph adj matrix
AM = G.generate_graph(json, dataset_path = data_root_path, loader = dataloader(), save_path = './dicegraph_results')

for class_ in infoclasses.keys():
	print(np.array(AM[class_]).shape)
	plt.clf()
	plt.imshow(AM[class_], cmap = plt.cm.RdBu, vmin = 0, vmax = 1)
	plt.colorbar()
	plt.savefig('./dicegraph_results/' + class_+'.png')
significance = G.node_significance(json, dataset_path = data_root_path, loader = dataloader(), save_path = './dicegraph_results')
pprint(significance)
	
