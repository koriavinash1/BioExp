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

from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm
from skimage.transform import resize as imresize

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure


class CausalGraph():
	"""
		class to generate causal 
	"""
	def __init__(self, model, weights_pth, metric, layer_names, max_clusters = None, classinfo=None):
		
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
		self.layers     = layer_names
		self.metric     = metric
		self.classinfo  = classinfo
		self.noutputs   = len(self.model.outputs)


	def get_layer_idx(self, layer_name):
		for idx, layer in enumerate(self.model.layers):
			if layer.name == layer_name:
				return idx
		

	def get_link(self, nodeA_info, nodeB_info, dataset_path, loader, max_samples = 1):
		"""
			get link between two nodes, nodeA, nodeB
			occlude at nodeA and observe changes in nodeB
			nodeA_info    : {'layer_name', 'layer_idxs'}
			nodeB_info    : {'layer_name', 'layer_idxs'}
		"""
		self.model.load_weights(self.weights, by_name = True)

		nodeA_idx   = self.get_layer_idx(nodeA_info['layer_name'])
		nodeA_idxs  = nodeA_info['layer_idxs']
		nodeB_idx   = self.get_layer_idx(nodeB_info['layer_name'])
		nodeB_idxs  = nodeB_info['layer_idxs']
                
                total_filters = np.arange(np.array(self.model.layers[nodeB_idx].get_weights())[0].shape[-1])
                test_filters  = np.delete(total_filters, nodeB_idxs)

                layer_weights = np.array(self.model.layers[node_idx].get_weights().copy())
                occluded_weights = layer_weights.copy()
                for j in test_filters:
                        occluded_weights[0][:,:,:,j] = 0
                        try:
                                occluded_weights[1][j] = 0
                        except: pass

                self.model.layers[nodeB_idx].set_weights(occluded_weights)
                model = Model(inputs = self.model.input, outputs=self.model.get_layer(nodeB_info['layer_name']).output)


                true_distributionB = []
		input_paths = os.listdir(dataset_path)
		for i in range(len(input_paths) if len(input_paths) < max_samples else max_samples):
			input_, label_ = loader(os.path.join(dataset_path, input_paths[i]), 
								os.path.join(dataset_path, 
								input_paths[i]).replace('mask', 'label').replace('labels', 'masks'))
			prediction = np.squeeze(self.model.predict(input_[None, ...]))




		for class_ in self.classinfo.keys():
			dice_json[class_] = np.mean(dice_json[class_])

		return dice_json


	def generate_graph(self, graph_info, dataset_path, dataset_path, dataloader, save_path = None, verbose = False, max_samples=10):
		layers   = graph_info['layer_name']
		concept_names = graph_info['concept_name']
		filter_idxs   = graph_info['feature_map_idxs']

		layer_names  = np.unique(layers)
                layer_names  = layer_names[np.argsort([int(idx.split('_')[-1]) for idx in layer_names])]

                node_ordering = []
                node_indexing = []

		for i in renage(len(layer_names)):
			node_ordering.extend(concept_names[layers == layer_names[i])
                        node_indexing.extend([i]*sum(layers == layer_names[i]))
                
                node_indexing = np.array(node_indexing)
                node_ordering = np.array(node_ordering)

                self.causal_graph = np.zeros((len(node_ordering), len(node_ordering))).astype('int')
                for ii, (idxi, nodei) in enumerate(zip(node_indexing, node_ordering)):
                        nodei_info = {'concept_name': nodei, 'layer_name': layers[concept_names == nodei], 'feature_map_idxs': filter_idxs[concept_names == nodei]}
                        for jj, (idxj, nodej) in enumerate(zip(node_indexing[node_indexing > idxi], node_ordering[node_indexing > idx])):
                                nodej_info = {'concept_name': nodej, 'layer_name': layers[concept_names == nodej], 'feature_map_idxs': filter_idxs[concept_names == nodei]}
                                self.causal_graph[ii, jj] = self.get_link(nodei_info,
                                                                            nodej_info,
                                                                            dataset_path = dataset_path,
                                                                            loader = dataloader,
                                                                            max_samples = max_samples)

		pass

	def perform_intervention(self):
		pass
