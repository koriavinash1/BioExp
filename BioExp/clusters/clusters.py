import matplotlib
matplotlib.use('Agg')
import keras
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


class Cluster():
	"""
	A class for conducting an cluster study on a trained keras model instance

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
	

	def get_distances(self, X, model, mode='l2'):
		"""
		"""
		distances = []
		weights = []
		children=model.children_

		dims = (X.shape[1],1)
		distCache = {}
		weightCache = {}
		for childs in children:
			c1 = X[childs[0]].reshape(dims)
			c2 = X[childs[1]].reshape(dims)
			c1Dist = 0
			c1W = 1
			c2Dist = 0
			c2W = 1
			if childs[0] in distCache.keys():
				c1Dist = distCache[childs[0]]
				c1W = weightCache[childs[0]]
			if childs[1] in distCache.keys():
				c2Dist = distCache[childs[1]]
				c2W = weightCache[childs[1]]
			d = np.linalg.norm(c1-c2)
			# d = np.squeeze(np.dot(c1.T, c2)/ (np.linalg.norm(c1)*np.linalg.norm(c2)))
			cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

			X = np.vstack((X,cc.T))

			newChild_id = X.shape[0]-1

			# How to deal with a higher level cluster merge with lower distance:
			if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
				added_dist = ((c1Dist**2+c2Dist**2)**0.5)
				dNew = (d**2 + added_dist**2)**0.5
			elif mode == 'max':  # If the previrous clusters had higher distance, use that one
				dNew = max(d,c1Dist,c2Dist)
			elif mode == 'cosine':
				dNew = np.squeeze(np.dot(c1Dist, c2Dist)/ (np.linalg.norm(c1Dist)*np.linalg.norm(c2Dist)))
			elif mode == 'actual':  # Plot the actual distance.
				dNew = d

			wNew = (c1W + c2W)
			distCache[newChild_id] = dNew
			weightCache[newChild_id] = wNew

			distances.append(dNew)
			weights.append(wNew)
		return distances, weights


	def plot_dendrogram(self, X, model, threshold=.7):
		"""
		"""

		# Create linkage matrix and then plot the dendrogram
		distance, weight = self.get_distances(X,model)
		linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)

		threshold = threshold*np.max(distance)
		
		sorted_   = linkage_matrix[np.argsort(distance)]
		splitnode = np.max(sorted_[sorted_[:, 2] > threshold][0, (0,1)])
		
		level     = np.log((-.5*splitnode)/(1.*X.shape[0]) + 1.)/np.log(.5)
		nclusters = int(np.round((1.*X.shape[0])/(2.**level))) 
		

		### New ....
		model = AgglomerativeClustering(n_clusters=nclusters).fit(X)
		distance, weight = self.get_distances(X, model)
		linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)
		labels = model.labels_
		
		print ("===========================", nclusters, threshold, np.max(distance), np.unique(labels), [sum(labels == i) for i in np.unique(labels)])
		# Plot the corresponding dendrogram

		return linkage_matrix, labels


	def get_clusters(self, threshold=0.5, save_path = None):
		"""
		Does clustering on feature space

		save_path  : path to save dendrogram image
		threshold  : fraction of max distance to cluster 
		"""

		layer_weights = np.array(self.model.layers[self.layer_idx].get_weights())

		shape = layer_weights[0].shape
		X = layer_weights[0].reshape(layer_weights[0].shape[-1], -1)
		position = np.linspace(0, X.shape[-1], X.shape[-1])
		X = X + position[None, :]
		model = AgglomerativeClustering().fit(X)
		
		# plot the top three levels of the dendrogram
		linkage_matrix, labels = self.plot_dendrogram(X, model, threshold = threshold)
		
		if save_path:
			os.makedirs(save_path, exist_ok=True)
			plt.figure(figsize=(20, 10))
			plt.title('Hierarchical Clustering Dendrogram')
			R = dendrogram(linkage_matrix, truncate_mode='level')
			plt.xlabel("Number of points in node (or index of point if no parenthesis).")
			plt.savefig(os.path.join(save_path, '{}_dendrogram.png'.format(self.layer)))
		
		return labels 






