import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from glob import glob
import sys
import os
sys.path.append('../../')

import pickle
from tqdm import tqdm
from BioExp.helpers import utils
from BioExp.spatial import ablation

from BioExp.spatial.losses import *
from BioExp.helpers.metrics import *
from BioExp.helpers import utils, radfeatures
from lucid.modelzoo.vision_base import Model
from BioExp.concept.feature import Feature_Visualizer
from keras.backend.tensorflow_backend import set_session

import argparse

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='feature study')
parser.add_argument('--seq', default='flair', type=str, help='mri sequence')
parser = parser.parse_args()
seq = parser.seq
print (seq)

data_root_path = '../../sample_vol/brats/'

layers_to_consider = ['conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12', 'conv2d_13', 'conv2d_14', 'conv2d_15', 'conv2d_16', 'conv2d_17', 'conv2d_18', 'conv2d_19', 'conv2d_20', 'conv2d_21']
input_name = 'input_1'

infoclasses = {}
for i in range(1): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3)
num_images = 10

model_pb_path = '../../../saved_models/model_{}/model.pb'.format(seq)	
model_path = '../../../saved_models/model_{}/model-archi.h5'.format(seq)
weights_path = '../../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)


model = load_model(model_path, 
		custom_objects={'gen_dice_loss': gen_dice_loss,
				'dice_whole_metric':dice_whole_metric,
				'dice_core_metric':dice_core_metric,
				'dice_en_metric':dice_en_metric})
print("Models Loaded")
n_classes = len(infoclasses)
metric = dice_label_coef

feature_maps = []
layers = []
classes = []

for layer_name in layers_to_consider:	
	print(layer_name)
	for i, file in tqdm(enumerate(glob(data_root_path +'*')[:num_images])):
		test_image, gt = utils.load_vol_brats(file, slicen=78)
		test_image = test_image[:, :, 0].reshape((1, 240, 240, 1))	

		A = ablation.Ablation(model, weights_path, metric, layer_name, test_image, gt, classes = infoclasses)
		if i == 0:
			df = A.ablate_filter(step = 1)
		else:
			df += A.ablate_filter(step = 1)
			
	df = df/(1. * num_images)
	save_path = 'results/Ablation/unet_{}/csv/'.format(seq, layer_name)
	os.makedirs(save_path, exist_ok=True)
	df.to_csv('results/Ablation/unet_{}/csv/{}.csv'.format(seq, layer_name))

	# list all featuremap dice greater than 0.1
	n_top = 5
	dice_matrix = df.values[:, 1:]
	dice_matrix = (dice_matrix - np.min(dice_matrix, axis=0))/(np.max(dice_matrix, axis=0) - np.min(dice_matrix, axis=0))
	dice_matrix = dice_matrix > 0.5
	feature_info, class_info = np.where(dice_matrix)
	index = np.arange(len(feature_info))
	np.random.shuffle(index)
	feature_info = feature_info[index[:n_top]]
	class_info = class_info[index[:n_top]]
	layers.extend([layer_name]*len(feature_info))
	feature_maps.extend(feature_info)
	classes.extend(class_info)


#########################################################################
# import ipdb
#ipdb.set_trace()
K.clear_session()

# Initialize a class which loads a Lucid Model Instance with the required parameters
from BioExp.helpers.pb_file_generation import generate_pb

if not os.path.exists(model_pb_path):
    print (model.summary())
    layer_name = 'conv2d_21'# str(input("Layer Name: "))
    generate_pb(model_path, layer_name, model_pb_path, weights_path)

input_name = 'input_1' 
class Load_Model(Model):
    model_path = model_pb_path
    image_shape = [None, 1, 240, 240]
    image_value_range = (0, 1)
    input_name = input_name


graph_def = tf.GraphDef()
with open(model_pb_path, "rb") as f:
    graph_def.ParseFromString(f.read())

# for node in graph_def.node:
#     print(node.name)


print ("==========================")
texture_maps = []
print (np.unique(classes))

# pdb.set_trace()
counter  = 0
save_pth = os.path.join('results/Ablation/unet_{}/lucid/'.format(seq))
os.makedirs(save_pth, exist_ok=True)

regularizer_params = {'L1':1e-5, 'rotate':10}

E = Feature_Visualizer(Load_Model, 
			savepath = save_pth, 
			regularizer_params = regularizer_params)

for layer_, feature_, class_ in zip(layers, feature_maps, classes):
    # if counter == 2: break
    # K.clear_session()

    print (layer_, feature_)
    # Initialize a Visualizer Instance
    texture_maps.append(E.run(layer = layer_, # + '_' + str(feature_), 
				channel = feature_, 
                                class_  = "class_"+str(class_),
                                transforms = True)) 
    counter += 1


json = {'textures': texture_maps, 
	'class_info': classes, 
	'features': feature_maps, 
	'layer_info': layers}

import pickle
pickle_path = os.path.join('results/Ablation/unet_{}/lucid/'.format(seq))
file_ = open(os.path.join(pickle_path, 'all_info'), 'wb')
pickle.dump(json, file_)

#########################################################################

# radiomic analysis
for ii, (tmap, _class_) in enumerate(zip(texture_maps, classes)):
    
	# create sitk object
	# ipdb.set_trace()		
	save_path = os.path.join('results/Ablation/unet_{}/RadiomicAnalysis/amaps/class_{}/{}'.format(seq, class_, ii))
	os.makedirs(save_path, exist_ok=True)
    
	tmps = np.array(tmap)
	print (tmps.shape)
	feat_extractor = radfeatures.ExtractRadiomicFeatures(tmps,
                                    save_path = save_path)

	df = feat_extractor.all_features()  
	print (df)

