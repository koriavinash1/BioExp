import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from glob import glob
import keras.backend as K
from keras.utils import np_utils
from tqdm import tqdm

class Ablation():

	def __init__(self, model, weights, metric, layer, test_image, gt, mode='whole'):
		
		self.model = model
		self.weights = weights
		self.metric = metric
		self.test_image = test_image
		self.layer = layer
		self.gt = gt
		self.mode = mode


	def ablate_filter(self, step):

		layer, _filter, class_list, value = [], [], [], []

		#classes = self.model.layers[-1].output.shape[-1]
		classes = 1

		filters_to_ablate = np.arange(0, self.model.layers[self.layer].get_weights()[0].shape[-1], step)
		            
		#print('Layer = %s' %self.model.layers[self.layer].name)
		self.model.load_weights(self.weights, by_name = True)

		#predicts each volume and save the results in np array
		prediction_unshaped = self.model.predict(self.test_image,batch_size=1,verbose=0)

		if self.mode == 'whole':

			for j in tqdm(filters_to_ablate):
				#print('Perturbed_Filter = %d' %j)
				self.model.load_weights(self.weights, by_name = True)
				layer_weights = np.array(self.model.layers[self.layer].get_weights())

				occluded_weights = layer_weights.copy()
				occluded_weights[0][:,:,:,j] = 0
				occluded_weights[1][j] = 0
				self.model.layers[self.layer].set_weights(occluded_weights)

				prediction_unshaped_occluded = self.model.predict(self.test_image,batch_size=1,verbose=0) 

				layer.append(self.layer)
				_filter.append(j)
				class_list.append('whole')
				value.append(self.metric(np_utils.to_categorical(self.gt, num_classes=4), 
					np_utils.to_categorical(prediction_unshaped.argmax(axis = -1), num_classes=4)) - self.metric(np_utils.to_categorical(self.gt, num_classes=4), 
					np_utils.to_categorical(prediction_unshaped_occluded.argmax(axis = -1), num_classes=4)))

				# sorted_index = np.argsort(np.array(value))
				# layer, _filter, class_list, value = layer[sorted_index], _filter[sorted_index], class_list[sorted_index], value[sorted_index]

			json = {'layer': layer, 'filter': _filter, 'class_list': class_list, 'value': value}

			#K.clear_session()		

			return(json)

		else:

			for _class in tqdm(range(classes)):

			    for j in tqdm(filters_to_ablate):
			        #print('Perturbed_Filter = %d' %j)
			        self.model.load_weights(self.weights, by_name = True)
			        layer_weights = np.array(self.model.layers[self.layer].get_weights())

			        occluded_weights = layer_weights.copy()
			        occluded_weights[0][:,:,:,j] = 0
			        occluded_weights[1][j] = 0
			        self.model.layers[self.layer].set_weights(occluded_weights)

			        prediction_unshaped_occluded = self.model.predict(self.test_image,batch_size=1,verbose=0) 

			        layer.append(self.layer)
			        _filter.append(j)
			        class_list.append(_class)
			        value.append(self.metric(np_utils.to_categorical(self.gt, num_classes=4), 
			        	np_utils.to_categorical(prediction_unshaped.argmax(axis = -1), num_classes=4), _class) - self.metric(np_utils.to_categorical(self.gt, num_classes=4), 
			        	np_utils.to_categorical(prediction_unshaped_occluded.argmax(axis = -1), num_classes=4), _class))

			    # sorted_index = np.argsort(np.array(value))
			    # layer, _filter, class_list, value = layer[sorted_index], _filter[sorted_index], class_list[sorted_index], value[sorted_index]

			json = {'layer': layer, 'filter': _filter, 'class_list': class_list, 'value': value}

			#K.clear_session()		

			return(json)


# if __name__ == '__main__':

# 	def dice_(y_true, y_pred):
# 	#computes the dice score on two tensors

# 		sum_p=K.sum(y_pred,axis=0)
# 		sum_r=K.sum(y_true,axis=0)
# 		sum_pr=K.sum(y_true * y_pred,axis=0)
# 		dice_numerator =2*sum_pr
# 		dice_denominator =sum_r+sum_p
# 		print(K.get_value(sum_pr), K.get_value(sum_p))
# 		dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
# 		return dice_score

# 	def metric(y_true, y_pred):
# 	#computes the dice for the whole tumor

# 		y_true_f = K.reshape(y_true,shape=(-1,4))
# 		y_pred_f = K.reshape(y_pred,shape=(-1,4))
# 		y_whole=K.sum(y_true_f[:,1:],axis=1)
# 		p_whole=K.sum(y_pred_f[:,1:],axis=1)
# 		dice_whole=dice_(y_whole,p_whole)
# 		return dice_whole

# 	def dice_label_metric(y_true, y_pred, label):
# 	#computes the dice for the enhancing region

# 		y_true_f = K.reshape(y_true,shape=(-1,4))
# 		y_pred_f = K.reshape(y_pred,shape=(-1,4))
# 		y_enh=y_true_f[:,label]
# 		p_enh=y_pred_f[:,label]
# 		dice_en=dice_(y_enh,p_enh)
# 		return dice_en

# 	data_root_path = '../sample_vol/'

# 	model_path = '/media/balaji/CamelyonProject/parth/saved_models/model_flair/model-archi.h5'
# 	weights_path = '/media/balaji/CamelyonProject/parth/saved_models/model_flair/model-wts-flair.hdf5'

# 	test_image, gt = utils.load_vol_brats('../sample_vol/Brats18_CBICA_ARZ_1', slicen=78)

# 	model = load_model(model_path, 
# 		custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,
# 		'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})

# 	test_image = test_image.reshape((1, 240, 240, 4))

# 	A = Ablation(model, weights_path, dice_label_metric, 16, test_image)

# 	print(A.ablate_filter(10))
