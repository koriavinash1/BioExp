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


import pandas as pd
from ..helpers.utils import *
from ..spatial.ablation import Ablate
from ..spatial.clusters import Cluster

from keras.models import Model
from keras.utils import np_utils
from skimage.transform import resize as imresize

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure



class ConceptGraph():
	"""
	A class for generating concept graph on a trained keras model instance
	"""     

	def __init__(self, model, weights_pth, metric, layer_names, max_clusters = None):
		
		"""
			model       : keras model architecture (keras.models.Model)
			weights_pth : saved weights path (str)
			metric      : metric to compare prediction with gt, for example dice, CE
			layer_name  : name of the layer which needs to be ablated
			test_img    : test image used for ablation
			max_clusters: maximum number of clusters per layer
		"""     

		self.model      = model
		self.weights    = weights_pth
		self.layers     = layer_names
		self.metric     = metric


	def get_layer_idx(self, layer_name):
		for idx, layer in enumerate(self.model.layers):
			if layer.name == layer_name:
				return idx

	def get_concepts(self, save_path):
		"""
			Define node and generates json map

			save_path : path to save json graph
		"""

		graph_info = {'concept_name': [], 'layer_name': [], 'feature_map_idxs': []}

		node = 1
		for layer_name in self.layers:
			C = Cluster(self.model, self.weights, layer_name)
			concepts = C.get_clusters(threshold = 0.5, save_path='cluster_results')

			for concept in np.unique(concepts):
				graph_info['concept_name'].append('Node_' + str(node))
				graph_info['layer_name'].append(layer_name)
				idxs = np.arange(len(concepts)).astype('int')[concepts == concept]
				graph_info['feature_map_idxs'].append(list(idxs))
				node += 1

		os.makedirs(json_path, exist_ok = True)
		with open(os.path.join(save_path, 'concept_graph.pickle'), 'wb') as f:
			pickle.dump(graph_info, f)

		return graph_info


	def generate_link(self, fmaps):
		pass
		
	def generate_fmaps(self, nodeA_info, nodeB_info, dataset_path, loader, save_path):
		"""
			get link between two nodes, nodeA, nodeB

			nodeA_info    : {'layer_name', 'layer_idxs'}
			nodeB_info    : {'layer_name', 'layer_idxs'}
		"""

		nodeA_layer = nodeA_info['layer_name']
		nodeA_idx   = self.get_layer_idx(nodeA_info['layer_name'])
		nodeA_idxs  = nodeA_info['layer_idxs']

		nodeB_layer = nodeB_info['layer_name']
		nodeB_idx   = self.get_layer_idx(nodeB_info['layer_name'])
		nodeB_idxs  = nodeB_info['layer_idxs']


		self.model.load_weights(self.weights, by_name = True)
		self.layer_weights = np.array(self.model.layers[nodeA_idx].get_weights())
		occluded_weights = self.layer_weights.copy()

		for j in nodeB_idxs:
			occluded_weights[0][:,:,:,j] = 0
			occluded_weights[1][j] = 0

		self.model.layers[nodeA_idx].set_weights(occluded_weights)
		if os.path.exists(os.path.join(save_path, 'A_{}_B_{}_fmaps.npy'.format(nodeA_info['concept_name'], nodeB_info['concept_name']))):
			fmaps = np.load(os.path.join(save_path, 'A_{}_B_{}_fmaps.npy'.format(nodeA_info['concept_name'], nodeB_info['concept_name']))) 

		else:
			fmaps = []
			input_paths = os.listdir(dataset_path)

			for i in range(len(input_paths) if len(input_paths) < 500 else 500):
				print ("[INFO: BioExp] Slice no {} -- Working on {}".format(self.layer_name, i))
				input_, label_ = loader(os.path.join(dataset_path, input_paths[i]), 
										os.path.join(dataset_path, 
													input_paths[i]).replace('mask', 'label').replace('labels', 'masks'))
				output = np.squeeze(self.model.predict(input_[None, ...]))
				fmaps.append(output)

			fmaps = np.array(fmaps)

			if not os.path.exists(save_path): 
				os.makedirs(save_path)

			np.save(os.path.join(save_path, 'A_{}_B_{}_fmaps.npy'.format(nodeA_info['concept_name'], nodeB_info['concept_name'])), fmaps)





