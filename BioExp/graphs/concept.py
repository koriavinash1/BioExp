import matplotlib
matplotlib.use('Agg')
import keras
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


class ConceptGraph():
	"""
	A class for generating concept graph on a trained keras model instance

	"""     


	def __init__(self, model, weights_pth, layer_name, max_clusters = None):
		
		"""
		model       : keras model architecture (keras.models.Model)
		weights_pth : saved weights path (str)
			metric      : metric to compare prediction with gt, for example dice, CE
			layer_name  : name of the layer which needs to be ablated
			test_img    : test image used for ablation
			max_clusters: maximum number of clusters
		"""     

		self.model = model
		self.weights = weights_pth
		self.layer = layer_name
		self.layer_idx = 0
		for idx, layer in enumerate(self.model.layers):
			if layer.name == self.layer:
				self.layer_idx = idx
