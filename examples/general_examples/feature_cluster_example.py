import sys
sys.path.append('../..')
import argparse
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
from glob import glob
from BioExp.helpers import utils
from BioExp.clusters import feature_clustering as clustering
import os

from keras.backend.tensorflow_backend import set_session
from BioExp.helpers.metrics import *
from BioExp.helpers.losses import *

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

	
parser = argparse.ArgumentParser(description='feature study')
parser.add_argument('--seq', default='flair', type=str, help='mri sequence')
parser = parser.parse_args()


seq_map = {'flair': 0, 't1': 1, 't2': 3, 't1c':2}
seq = parser.seq


print (seq)
model_path        = '../../saved_models/model_{}/model-archi.h5'.format(seq)
weights_path      = '../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)


layers_to_consider = ['conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5']
model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
	                                'dice_whole_metric':dice_whole_metric,
	                                'dice_core_metric':dice_core_metric,
	                                'dice_en_metric':dice_en_metric})
model.load_weights(weights_path)



for layer_name in layers_to_consider:
    C = clustering.Cluster(model, weights_path, layer_name, method = 'gmm', max_clusters=5)
    labels = C.get_clusters(save_path='cluster_results')
    print (labels)
    
    C.plot_weights(n = 5, save_path = 'model_weights')
    C.plot_features(projection_dim = 2, save_path = 'extracted_features')
    C.plot_features(projection_dim = 3, save_path = 'extracted_features')
    C.plot_clusters(save_path = 'cluster_results')
