import matplotlib
matplotlib.use('Agg')
import keras
import numpy as np
import tensorflow as tf
import os
import pdb
import cv2 
import pickle
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import pandas as pd
from ..helpers.utils import *
from ..spatial.ablation import Ablate

from keras.models import Model


class DeltaGraph():
	"""
	A class for generating concept graph on a trained keras model instance
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


	def significance_test(self, concept_info, dataset_path, loader, nmontecarlo = 10, max_samples = 1):
		"""
			test significance of each concepts

			concept: {'concept_name', layer_name', 'filter_idxs'}
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

		for class_ in self.classinfo.keys():
			dice_json[class_] = np.mean(dice_json[class_])

		return dice_json 
		

	def get_link(self, nodeA_info, nodeB_info, dataset_path, loader, save_path, max_samples = 1):
		"""
			get link between two nodes, nodeA, nodeB
			occlude at nodeA and observe changes in nodeB

			nodeA_info    : {'layer_name', 'layer_idxs'}
			nodeB_info    : {'layer_name', 'layer_idxs'}
		"""
		self.modelcopy.load_weights(self.weights, by_name = True)
		self.model.load_weights(self.weights, by_name = True)

		nodeA_idx   = self.get_layer_idx(nodeA_info['layer_name'])
		nodeA_idxs  = nodeA_info['filter_idxs']

		nodeB_idx   = self.get_layer_idx(nodeB_info['layer_name'])
		nodeB_idxs  = nodeB_info['filter_idxs']


		layer_weights = np.array(self.modelcopy.layers[nodeA_idx].get_weights())
		occluded_weights = layer_weights.copy()
		for j in nodeA_idxs:
			occluded_weights[0][:,:,:,j] = 0
			occluded_weights[1][j] = 0
		self.modelcopy.layers[nodeA_idx].set_weights(occluded_weights)

		layer_weights = np.array(self.modelcopy.layers[nodeB_idx].get_weights())
		occluded_weights = layer_weights.copy()

		for j in nodeB_idxs:
			occluded_weights[0][:,:,:,j] = 0
			occluded_weights[1][j] = 0
		self.modelcopy.layers[nodeB_idx].set_weights(occluded_weights)

		dice_json = {}
		for class_ in self.classinfo.keys():
			dice_json[class_] = []

		input_paths = os.listdir(dataset_path)
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


		for class_ in self.classinfo.keys():
			dice_json[class_] = np.mean(dice_json[class_])

		return dice_json


	def generate_graph(self, graph_info, 
							dataset_path = None, 
							loader = None, 
							save_path=None, 
							max_samples = 1, 
							nmontecarlo = 10):
		"""
			generates graph adj matrix for computation

			graph_info: [{'concept_name', 'layer_name', 'filter_idxs'}]
			save_path : graph_path or path to save graph
		"""

		if os.path.exists(os.path.join(save_path, 'concept_adj_matrix.pickle')):
			with open(os.path.join(save_path, 'concept_adj_matrix.pickle'), 'rb') as f:
				AM = pickle.load(f) 

		else:
			nodes = len(graph_info)

			AM = {}
			for class_ in self.classinfo.keys():
				AM[class_] = []

			for nodeA in tqdm(range(nodes)):
				AM_row = {}
				for class_ in self.classinfo.keys():
					AM_row[class_] = []


				for nodeB in tqdm(range(nodes)):
					if nodeA == nodeB:
						node_info = graph_info[nodeA]

						significance_dice = self.significance_test(node_info,
														dataset_path, loader, 
														nmontecarlo = nmontecarlo,
														max_samples = max_samples )
					nodeA_info = graph_info[nodeA]
					nodeB_info = graph_info[nodeB]
					link = self.get_link(nodeA_info, nodeB_info,
													dataset_path = dataset_path, 
													loader = loader, 
													save_path = save_path,
													max_samples = max_samples)
					for class_ in self.classinfo.keys():
						AM_row[class_].append(link[class_])
				
				for class_ in self.classinfo.keys():
					AM[class_].append(AM_row[class_])

			with open(os.path.join(save_path, 'concept_adj_matrix.pickle'), 'wb') as f:
				pickle.dump(AM, f) 

		return AM

	def node_significance(self, graph_info, dataset_path = None, loader = None, save_path=None, max_samples = 1, nmontecarlo = 10):
		"""
			generates graph adj matrix for computation

			graph_info: {'concept_name', 'layer_name', 'feature_map_idxs'}
			save_path : graph_path or path to save graph
		"""

		if os.path.exists(os.path.join(save_path, 'significance_info.pickle')):
			with open(os.path.join(save_path, 'significance_info.pickle'), 'rb') as f:
				significance = pickle.load(f) 

		else:
			nodes = len(graph_info)

			significance = {}

			for i in range(nodes):
				node_info = graph_info[i] 
				significance_dice = self.significance_test(node_info,
												dataset_path, loader, 
												nmontecarlo = nmontecarlo,
												max_samples = max_samples )
				
				significance[graph_info[i]['concept_name']] = significance_dice
			with open(os.path.join(save_path, 'significance_info.pickle'), 'wb') as f:
				pickle.dump(significance, f) 

		return significance
