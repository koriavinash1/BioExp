import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from losses import *
from glob import glob

class Ablation():

	def __init__(self, model, weights, metric, layer, test_image):
		
		self.model = model
		self.weights = weights
		self.metric = metric
		self.test_image = test_image
		self.layer = layer


	def ablate_filter(self, step):

		class_wise_scores = {}

		classes = self.model.layers[-1].output.shape[-1]

		filters_to_ablate = np.arange(0, self.model.layers[self.layer].get_weights()[0].shape[-1], step)
		            
		print('Layer = %s' %self.model.layers[self.layer].name)
		self.model.load_weights(self.weights, by_name = True)

		#predicts each volume and save the results in np array
		prediction_unshaped = self.model.predict(self.test_image,batch_size=1,verbose=0)

		for _class in range(classes):

		    intervention_dict = {}

		    weights = np.array(self.model.layers[self.layer].get_weights())

		    for j in filters_to_ablate:
		        #print('Perturbed_Filter = %d' %j)
		        self.model.load_weights(self.weights, by_name = True)

		        weights = np.array(self.model.layers[self.layer].get_weights())

		        occluded_weights = weights.copy()
		        occluded_weights[0][:,:,:,j] = 0
		        occluded_weights[1][j] = 0
		        self.model.layers[self.layer].set_weights(occluded_weights)
		        
		        prediction_unshaped_occluded = self.model.predict(test_image,batch_size=1,verbose=0) 
		        
		        intervention_dict[(self.layer,j)] = 1-K.get_value(self.metric(prediction_unshaped, prediction_unshaped_occluded, _class))

		    sorted_dict = sorted(intervention_dict.items(), key=lambda x:x[1], reverse = True)

		    class_wise_scores[_class] = sorted_dict

		return(class_wise_scores)


if __name__ == '__main__':

	def dice_(y_true, y_pred):
	#computes the dice score on two tensors

		sum_p=K.sum(y_pred,axis=0)
		sum_r=K.sum(y_true,axis=0)
		sum_pr=K.sum(y_true * y_pred,axis=0)
		dice_numerator =2*sum_pr
		dice_denominator =sum_r+sum_p
		print(K.get_value(sum_pr), K.get_value(sum_p))
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

	path = glob('/media/parth/DATA/brats_slices/_train/patches/*.npy')

	model = load_model('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/og_pipeline/ResUnet.h5', 
		custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,
		'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})

	weights = '/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/og_pipeline/ResUnet.40_0.559.hdf5'

	test_image = np.load('/home/parth/Downloads/tumor_test.npy')

	test_image = test_image.reshape((1, 240, 240, 4))

	A = Ablation(model, weights, dice_label_metric, 16, test_image)

	print(A.ablate_filter(10))