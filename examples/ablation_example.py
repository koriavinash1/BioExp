import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from glob import glob
import sys
import os
sys.path.append('..')
from BioExp.helpers import utils
from BioExp.spatial import ablation
#from BioExp.helpers.losses import *
from BioExp.helpers.losses import *
from BioExp.helpers.metrics import *
import pickle
from lucid.modelzoo.vision_base import Model
from BioExp.concept.feature import Feature_Visualizer
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

seq = 'flair'


data_root_path = '../sample_vol/'

seq_to_consider = ['flair', 't1c', 't2', 't1']


for seq in seq_to_consider:

	model_pb_path = '../../saved_models/model_{}/model.pb'.format(seq)	
	model_path = '../../saved_models/model_{}/model-archi.h5'.format(seq)
	weights_path = '../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)
	mode = 'label'
	model = load_model(model_path, 
			custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,
			'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})

	for layer in range(0, 59):

		if mode == 'whole': 
			metric = dice_whole_coef
			n_classes = 1
		else:
			metric = dice_label_coef
			n_classes=4

		if 'conv2d' in model.layers[layer].name:	
			print(model.layers[layer].name)
			for file in tqdm(glob(data_root_path +'*')[:10]):

				test_image, gt = utils.load_vol_brats(file, slicen=78)

				test_image = test_image[:, :, 0].reshape((1, 240, 240, 1))	

				A = ablation.Ablation(model, weights_path, metric, layer, test_image, gt, mode=mode)

				ablation_dict = A.ablate_filter(1)

				try:
					values = pd.concat([values, pd.DataFrame(ablation_dict['value'])], axis=1)	
				except:
					values = pd.DataFrame(ablation_dict['value'], columns = ['value'])


			mean_value = values.mean(axis=1)

			for key in ablation_dict.keys():
				if key != 'value':
					try:
						layer_df = pd.concat([layer_df, pd.DataFrame(ablation_dict[key], columns = [key])], axis=1)	
					except:
						layer_df = pd.DataFrame(ablation_dict[key], columns = [key])

			layer_df = pd.concat([layer_df, mean_value.rename('value')], axis=1)	

			sorted_df = layer_df.sort_values(['class_list', 'value'], ascending=[True, False])

			for i in range(n_classes):
				save_path = '../results/Ablation/unet_{}/'.format(seq) + model.layers[layer].name
				os.makedirs(save_path, exist_ok=True)
				if mode == 'whole':
					class_df = sorted_df
					class_df.to_csv(save_path +'/class_{}.csv'.format('whole'))
				else:
					for i in range(4):
						class_df = sorted_df.loc[sorted_df['class_list'] == i]
						class_df.to_csv(save_path +'/class_{}.csv'.format(i))

			del values, layer_df, mean_value
# print(sorted_df['class_list'], sorted_df['value'])

# K.clear_session()
# # Initialize a class which loads a Lucid Model Instance with the required parameters
# from BioExp.helpers.pb_file_generation import generate_pb

# if not os.path.exists(model_pb_path):
#     print (model.summary())
#     layer_name = 'conv2d_21'# str(input("Layer Name: "))
#     generate_pb(model_path, layer_name, model_pb_path, weights_path)

# input_name = 'input_1' #str(input("Input Name: "))
# class Load_Model(Model):
#     model_path = model_pb_path
#     image_shape = [None, 1, 240, 240]
#     image_value_range = (0, 1)
#     input_name = input_name


# graph_def = tf.GraphDef()
# with open(model_pb_path, "rb") as f:
#     graph_def.ParseFromString(f.read())

# texture_maps = []

# counter  = 0
# for layer_, feature_, class_ in zip(sorted_df['layer'], sorted_df['filter'], sorted_df['class_list']):
#     # if counter == 2: break
#     K.clear_session()
    
#     # Run the Visualizer
#     print (layer_, feature_)
#     # Initialize a Visualizer Instance
#     save_pth = '/media/parth/DATA/datasets/BioExp_results/lucid/unet_{}/ablation/'.format(seq)
#     os.makedirs(save_pth, exist_ok=True)
#     E = Feature_Visualizer(Load_Model, savepath = save_pth)
#     texture_maps.append(E.run(layer = model.layers[layer].name, # + '_' + str(feature_), 
# 						 channel = feature_, transforms = True)) 
#     counter += 1


# json = {'texture':texture_maps, 'class':list(sorted_df['class_list']), 'filter':list(sorted_df['filter']), 'layer':layer, 'importance':list(sorted_df['value'])}


# pickle_path = '/media/parth/DATA/datasets/BioExp_results/lucid/unet_{}/ablation/'.format(seq)
# os.makedirs(pickle_path, exist_ok=True)
# file_ = open(os.path.join(pickle_path, 'all_info'), 'wb')
# pickle.dump(json, file_)
