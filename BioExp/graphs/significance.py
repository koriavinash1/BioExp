import keras
import numpy as np
import tensorflow as tf
import os
import pdb
import cv2 
import pickle


import pandas as pd
from ..helpers.utils import *

from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm


class SignificanceTester():
	"""
	A class for testing significance of each concepts generated in a trained keras model instance
	"""     

	def __init__(self, model, weights_pth, metric, classinfo=None):
		
		"""
			model       : keras model architecture (keras.models.Model)
			weights_pth : saved weights path (str)
			metric      : metric to compare prediction with gt, for example dice, CE
			layer_name  : name of the layer which needs to be ablated
			test_img    : test image used for ablation
			max_clusters: maximum number of clusters per layer
		"""     

		self.model      = model
		self.modelcopy  = keras.models.clone_model(self.model)
		self.weights    = weights_pth
		self.metric     = metric
		self.classinfo  = classinfo
		self.noutputs   = len(self.model.outputs)
	

	def get_layer_idx(self, layer_name):
		for idx, layer in enumerate(self.model.layers):
			if layer.name == layer_name:

				return idx

	def node_significance(self, concept_info, dataset_path, loader, nmontecarlo = 10, max_samples = 1):
		"""
			test significance of each concepts
			concept: {'layer_name', 'filter_idxs'}
		"""
		
		self.model.load_weights(self.weights, by_name = True)
		node_idx  = self.get_layer_idx(concept_info['layer_name'])
		node_idxs = concept_info['filter_idxs']

		total_filters = np.arange(np.array(self.model.layers[node_idx].get_weights())[0].shape[-1])
		nfilters      = len(node_idxs)
		test_filters  = np.delete(total_filters, node_idxs)
		if len(test_filters) < nfilters:
			print("Huge cluster size, may not be significant, cluster size: {}, total data size: {}".format(nfilters, len(test_filters)))
			return False

		input_paths = os.listdir(dataset_path)
		dice_json = {}
		for class_ in self.classinfo.keys():
			dice_json[class_] = []
		dice_json['IG'] = [] # information gain


		for _ in range(nmontecarlo):
			np.random.shuffle(test_filters)
			self.modelcopy.load_weights(self.weights, by_name = True)
			layer_weights = np.array(self.modelcopy.layers[node_idx].get_weights())
			occluded_weights = layer_weights.copy()

			for j in test_filters[:nfilters]:
				occluded_weights[0][:,:,:,j] = 0
				occluded_weights[1][j] = 0

			self.modelcopy.layers[node_idx].set_weights(occluded_weights)
			for i in range(len(input_paths) if len(input_paths) < max_samples else max_samples):
				input_, label_ = loader(os.path.join(dataset_path, input_paths[i]), 
									os.path.join(dataset_path, 
									input_paths[i]).replace('mask', 'label').replace('labels', 'masks'))
				prediction_occluded = np.squeeze(self.modelcopy.predict(input_[None, ...]))
				prediction = np.squeeze(self.model.predict(input_[None, ...]))
				
				idx = 0
				if self.noutputs > 1:
					for ii in range(self.noutputs):
						if prediction[ii] == self.nclasses:
							idx = ii 
							break;

				for class_ in self.classinfo.keys():
					if self.noutputs > 1:
						dice_json[class_].append(self.metric(label_, prediction[idx].argmax(axis = -1), self.classinfo[class_]) - 
									self.metric(label_, prediction_occluded[idx].argmax(axis = -1), self.classinfo[class_]))
					else:
						dice_json[class_].append(self.metric(label_, prediction.argmax(axis = -1), self.classinfo[class_]) - 
									self.metric(label_, prediction_occluded.argmax(axis = -1), self.classinfo[class_]))
				if self.noutputs > 1:
					dice_json['IG'].append(np.mean(-prediction_occluded[idx]*np.log2(prediction_occluded[idx]) + prediction[idx]*np.log(prediction[idx])))
				else:
					dice_json['IG'].append(np.mean(-prediction_occluded*np.log2(prediction_occluded) + prediction*np.log(prediction))) 


		for class_ in self.classinfo.keys():
			dice_json[class_] = np.mean(dice_json[class_])
		
		dice_json['IG'] = np.mean(dice_json['IG'])
		return dice_json 

	def graph_significance(self, graph_info, dataset_path = None, loader = None, save_path=None, max_samples = 1, nmontecarlo = 10):
		"""
			generates graph adj matrix for computation
			graph_info: {'concept_name', 'layer_name', 'feature_map_idxs'}
			save_path : graph_path or path to save graph
		"""

		if os.path.exists(os.path.join(save_path, 'significance_info.pickle')):
			with open(os.path.join(save_path, 'significance_info.pickle'), 'rb') as f:
				significance = pickle.load(f) 

		else:
			nodes = graph_info['concept_name']

			significance = {}

			for i, node in enumerate(nodes):
				node_info = {'layer_name': graph_info['layer_name'][i], 
								'filter_idxs':  graph_info['feature_map_idxs'][i]}
				significance_dice = self.node_significance(node_info,
										dataset_path, loader, 
										nmontecarlo = nmontecarlo,
										max_samples = max_samples )
				
				significance[node] = significance_dice
			with open(os.path.join(save_path, 'significance_info.pickle'), 'wb') as f:
				pickle.dump(significance, f) 

		return significance
