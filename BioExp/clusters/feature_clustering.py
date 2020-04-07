import matplotlib
matplotlib.use('Agg')
import keras
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


class Cluster():
    """
    A class for conducting an cluster study on a trained keras model instance
    """     


    def __init__(self, model, weights_pth, layer_name, max_clusters = None, method = None):
        
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
        self.max_clusters = max_clusters
        self.method = method
        self.layer_idx = 0
        for idx, layer in enumerate(self.model.layers):
            if layer.name == self.layer:
                self.layer_idx = idx
    

    def orientation_features(self, x):
        """
        """

        return feature

    def energy_features(self, x):
        """
        """

        return feature

    def other_features(self, x):
        """
        """

        return feature

    def other_features(self, x):
        """

        """
        return feature


    def get_features(self, wts):
        """
        wts: shape(k, k, in_c, out_c)

        """
        nfeatures = wts.shape[-1]
        features = []
        for i in nfeatures:
            feature = []
            feature.extend(self.orientation_features(wts[:, :, :, i]))
            feature.extend(self.energy_features(wts[:, :, :, i]))
            feature.extend(self.statistical_features(wts[:, :, :, i]))
            feature.extend(self.other_features(wts[:, :, :, i]))
            features.append(feature)

        return np.array(features)


    def GMM(self, X):
        """

        """
        return model


    def Agglomerative(self, X):
        """

        """
        return model


    def spectral(self, X):
        """

        """
        return model


    def kmeans(self, X):
        """
        """

        return model

    
    def get_cluster(self, X):
        """
        """

        self.model = None



    def plot_features(self, x, projection_dim = 2, label = None, save_path = None):
        """

        dim x: z x nfeatures
        projection_dim: feature rescale to .
        save_path: path to save plot
        """ 
        pca = PCA(n_components=projection_dim)
        pca.fit(x)
        X = pca.transform(x)

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()

        if projection_dim == 3:
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
            plt.cla()
            if not label: ax.scatter(X, cmap=plt.cm.nipy_spectral, edgecolor='k')
            else :
                for i in np.unique(label):
                    ax.scatter(X[label == i], c=i, cmap=plt.cm.nipy_spectral, edgecolor='k')
        elif projection_dim == 2:
            if not label: plt.scatter(X, cmap=plt.cm.nipy_spectral, edgecolor='k')
            else:
                for i in np.unique(label):
                    plt.scatter(X[label == i], c=i, cmap=plt.cm.nipy_spectral, edgecolor='k')
        else:
            raise ValueError("Projection dimension > 3, cannot be ploted")

        if not save_path:
            plt.show()
        else:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(save_path)

    
    def plot_clusters(self, x, save_path = None):
        """
        dim x: z x nfeatures
        save_path: path to save plot
        """
        try:
            label = self.model.predict(x)
        except:
            raise ValueError("Model not generated yet, First run get_cluster, attr")

        self.plot_features(x, 2, label, os.path.join(save_path, 'layer_{}_2d.png'.format(self.layer_idx)))
        self.plot_features(x, 3, label, os.path.join(save_path, 'layer_{}_3d.png'.format(self.layer_idx)))
        
