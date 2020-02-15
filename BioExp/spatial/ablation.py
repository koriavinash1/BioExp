import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import keras
import numpy as np
import tensorflow as tf
from glob import glob
import pandas as pd
from tqdm import tqdm
import keras.backend as K
from keras.utils import np_utils
from keras.models import load_model


class Ablate():
	"""
	A class for conducting an ablation study on a trained keras model instance
	
	"""		


	def __init__(self, model, weights_pth, metric, layer_name, test_image, gt, classes, nclasses=4, image_name=None):
		
		"""
		model       : keras model architecture (keras.models.Model)
		weights_pth : saved weights path (str)
                metric      : metric to compare prediction with gt, for example dice, CE
                layer_name  : name of the layer which needs to be ablated
                test_img    : test image used for ablation
                gt          : ground truth for comparision
                classes     : class informatiton which needs to be considered, class label as 
				key and corresponding required values 
				in a tuple: {'class1': (1,), 'whole': (1,2,3)}
                nclasses    : number of unique classes in gt
		"""		

		self.model = model
		self.weights = weights_pth
		self.metric = metric
		self.test_image = test_image
		self.layer = layer_name
		self.gt = gt
		self.classinfo = classes
		self.nclasses = nclasses
		self.layer_idx = 0
		self.image_name = image_name
		for idx, layer in enumerate(self.model.layers):
			if layer.name == self.layer:
				self.layer_idx = idx
		
		
		self.noutputs = len(self.model.outputs)
		self.model.load_weights(self.weights, by_name = True)
		self.layer_weights = np.array(self.model.layers[self.layer_idx].get_weights())
		self.filter_shape = self.layer_weights[0].shape


	def ablate_filters(self, filters_to_ablate = None, concept = 'random', step = None, save_path=None, verbose=1):
		"""
		Drops individual weights from the model, makes the prediction for the test image,
		and calculates the difference in the evaluation metric as compared to the non-
		ablated case. For example, for a layer with a weight matrix of shape 3x3x64, 
		individual 3x3 matrices are zeroed out at the interval given by the step argument.
		
		Arguments:
		step: The interval at which to drop weights
		Outputs: A dataframe containing the importance scores for each individual weight matrix in the layer
		"""
		
		if (step == None) and (filters_to_ablate == None):
			raise ValueError("step or filters_to_ablate is required")
		
		if filters_to_ablate == None:
			filters_to_ablate = np.arange(0, self.filter_shape[-1], step)
						
		#predicts each volume and save the results in np array
		prediction_unshaped, og_rec = self.model.predict(self.test_image, batch_size=1, verbose=verbose)


		dice_json = {}
		dice_json['concept'] = []
		for class_ in self.classinfo.keys():
			dice_json[class_] = []
			dice_json[class_] = []

		occluded_weights = self.layer_weights.copy()

		for j in (filters_to_ablate):
			occluded_weights[0][:,:,:,j] = 0
			occluded_weights[1][j] = 0

		self.model.layers[self.layer_idx].set_weights(occluded_weights)			
		prediction_unshaped_occluded, ab_rec = self.model.predict(self.test_image,batch_size=1, verbose=0)

		dice_json['concept'].append('actual_' + str(concept))
		dice_json['concept'].append('ablated_' + str(concept))

		for class_ in self.classinfo.keys():
			if self.noutputs == 1:
				dice_json[class_].append(self.metric(self.gt, prediction_unshaped.argmax(axis = -1), self.classinfo[class_]))
				dice_json[class_].append(self.metric(self.gt, prediction_unshaped_occluded.argmax(axis = -1), self.classinfo[class_]))
			else:
				for ii in range(self.noutputs):
					if prediction_unshaped[ii].shape[-1] == self.nclasses:
						idx = ii
				dice_json[class_].append(self.metric(self.gt, prediction_unshaped[idx].argmax(axis = -1), self.classinfo[class_]))
				dice_json[class_].append(self.metric(self.gt, prediction_unshaped_occluded[idx].argmax(axis = -1), self.classinfo[class_]))

		if not (save_path == None):
			if self.noutputs > 1:
				plt.subplot(1,2*self.noutputs + 2, 1)
				# plt.imshow(np.squeeze(self.test_image))
				plt.imshow(np.squeeze(self.test_image), alpha=1)

				plt.subplot(1,2*self.noutputs + 2, 2)
				# plt.imshow(np.squeeze(self.test_image))
				plt.imshow(np.squeeze(self.gt), alpha=1)
				for ii in range(self.noutputs):
					plt.subplot(1, 2*self.noutputs + 2, ii + 3)
					img = prediction_unshaped[ii]
					if img.shape[-1] == self.nclasses:
						img = img.argmax(axis = -1)
					plt.imshow(np.squeeze(img))
					plt.title('unoccluded')
				for ii in range(self.noutputs):
					plt.subplot(1, 2*self.noutputs + 2, self.noutputs + ii + 3)
					img = prediction_unshaped_occluded[ii]
					if img.shape[-1] == self.nclasses:
						img = img.argmax(axis = -1)
					plt.imshow(np.squeeze(img))
					plt.title('occluded')
			else:
				plt.subplot(1,4,1)
				# plt.imshow(np.squeeze(self.test_image))
				plt.imshow(np.squeeze(self.test_image), alpha=1)
				plt.subplot(1,4,2)
				# plt.imshow(np.squeeze(self.test_image))
				plt.imshow(np.squeeze(self.gt), alpha=1)
				plt.subplot(1,4,3)
				# plt.imshow(np.squeeze(self.test_image))
				plt.imshow(np.squeeze(prediction_unshaped.argmax(axis = -1)), alpha=1)
				plt.subplot(1,4,4)
				# plt.imshow(np.squeeze(self.test_image))
				plt.imshow(np.squeeze(prediction_unshaped_occluded.argmax(axis = -1)), alpha=1)
			plt.savefig(os.path.join(save_path, 'image_{}_{}_concept_{}.png'.format(self.image_name, self.layer, concept)))
		df = pd.DataFrame(dice_json)
		return df
