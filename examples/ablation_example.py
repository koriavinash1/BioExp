import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from glob import glob
import sys
import os
sys.path.append('..')
from BioExp.helpers import utils
from BioExp.spatial import ablation
from losses import *

seq = 'flair'
model_pb_path = '../../saved_models/model_{}/model.pb'.format(seq)
data_root_path = '../../slices/val/patches/'
model_path = '../../saved_models/model_{}/model-archi.h5'.format(seq)
weights_path = '../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)


def dice_(y_true, y_pred):
#computes the dice score on two tensors

	sum_p=K.sum(y_pred,axis=0)
	sum_r=K.sum(y_true,axis=0)
	sum_pr=K.sum(y_true * y_pred,axis=0)
	dice_numerator =2*sum_pr
	dice_denominator =sum_r+sum_p
	#print(K.get_value(2*sum_pr), K.get_value(sum_p)+K.get_value(sum_r))
	dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
	return dice_score

def metric(y_true, y_pred):
#computes the dice for the whole tumor

	y_true_f = K.reshape(y_true,shape=(-1,4))
	y_pred_f = K.reshape(y_pred,shape=(-1,4))
	y_whole=K.sum(y_true_f[:,1:],axis=1)
	p_whole=K.sum(y_pred_f[:,1:],axis=1)
	dice_whole=dice_(y_whole,p_whole)
	return dice_whole

def dice_label_metric(y_true, y_pred, label):
#computes the dice for the enhancing region
	
	y_true_f = K.reshape(y_true,shape=(-1,4))
	y_pred_f = K.reshape(y_pred,shape=(-1,4))
	y_enh=y_true_f[:,label]
	p_enh=y_pred_f[:,label]
	dice_en=dice_(y_enh,p_enh)
	return dice_en

data_root_path = '../sample_vol/'

model_path = '/media/balaji/CamelyonProject/parth/saved_models/model_flair/model-archi.h5'
weights_path = '/media/balaji/CamelyonProject/parth/saved_models/model_flair/model-wts-flair.hdf5'

test_image, gt = utils.load_vol_brats('../sample_vol/Brats18_CBICA_AQT_1', slicen=78)

model = load_model(model_path, 
	custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,
	'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})

test_image = test_image[:, :, 0].reshape((1, 240, 240, 1))	

A = ablation.Ablation(model, weights_path, dice_label_metric, 16, test_image, gt)

ablation_dict = A.ablate_filter(1)

for item in ablation_dict.values():
	print(item[:5])


# for channel_list in ablation_dict.values():
# 	for item in channel_list



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
# for node in graph_def.node:
#     print(node.name)


# print ("==========================")
# texture_maps = []
# print (np.unique(classes))

# counter  = 0
# for layer_, feature_, class_ in zip(layers, feature_maps, classes):
#     # if counter == 2: break
#     K.clear_session()
    
#     # Run the Visualizer
#     print (layer_, feature_)
#     # Initialize a Visualizer Instance
#     save_pth = '../results/lucid/unet_{}/'.format(seq)
#     os.makedirs(save_pth, exist_ok=True)
#     E = Feature_Visualizer(Load_Model, savepath = save_pth)
#     texture_maps.append(E.run(layer = layer_, # + '_' + str(feature_), 
# 						 channel = feature_)) 
#     counter += 1