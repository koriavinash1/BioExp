import matplotlib
matplotlib.use('Agg')
import keras
import numpy as np
import tensorflow as tf
import os, pickle
import SimpleITK as sitk
from radiomics.shape2D import RadiomicsShape2D
from radiomics.firstorder import RadiomicsFirstOrder
from radiomics.glcm import RadiomicsGLCM
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
        self.layer_name = layer_name
        self.max_clusters = max_clusters
        self.method = method
        self.layer_idx = 0
        for idx, layer in enumerate(self.model.layers):
            if layer.name == self.layer_name:
                self.layer_idx = idx
        self.weights = np.array(self.model.layers[self.layer_idx].get_weights())[0]
        # self.features = self.get_features(self.weights)
        self.features = self.flatten(self. weights)


    def normalize(self, x):
        """
        """
        norm01 = (x - np.min(x, axis=-1)[..., None])/(np.max(x, axis=-1)[..., None] - np.min(x, axis=-1)[..., None])
        return norm01


    def extract_features(self, x, mask, function):
        imr = sitk.GetImageFromArray
        try:
            features = function(imr(x), imr(mask))
            features.enableAllFeatures()
            features.execute()
            return np.array(list(features.featureValues.values()))
        except:
            return 0


    def orientation_features(self, x):
        """
        x: dim k x k x in_c
        """
        feature = self.extract_features(x[:,:,0], np.ones_like(x[:,:,0]), RadiomicsShape2D)
        for wt in range(1, x.shape[-1]):
            feature += self.extract_features(x[:,:,wt], np.ones_like(x[:,:,wt]), RadiomicsShape2D)
        feature /= x.shape[-1]
        return feature


    def statistical_features(self, x):
        """

        """
        feature = self.extract_features(x[:,:,0], np.ones_like(x[:,:,0]), RadiomicsFirstOrder)
        for wt in range(1, x.shape[-1]):
            feature += self.extract_features(x[:,:,wt], np.ones_like(x[:,:,wt]), RadiomicsFirstOrder)
        feature /= x.shape[-1]
        return feature


    def other_features(self, x):
        """

        """
        feature = self.extract_features(x[:,:,0], np.ones_like(x[:,:,0]), RadiomicsGLCM)
        for wt in range(1, x.shape[-1]):
            feature += self.extract_features(x[:,:,wt], np.ones_like(x[:,:,wt]), RadiomicsGLCM)
        feature /= x.shape[-1]
        return feature


    def get_features(self, wts):
        """
        wts: shape(k, k, in_c, out_c)

        """
        wts = self.normalize(wts)
        nfeatures = wts.shape[-1]
        features = []
        for i in range(nfeatures):
            feature = []
            # feature.extend(self.orientation_features(wts[:, :, :, i]))
            feature.extend(self.statistical_features(wts[:, :, :, i]))
            # feature.extend(self.other_features(wts[:, :, :, i]))
            features.append(feature)

        features = np.array(features)
        print ("Extracted feature dimension: {}".format(features.shape))
        return features



    def flatten(self, wts):
        """
        """
        wts = self.normalize(wts)
        features = wts.reshape(-1, wts.shape[-1]).T
        print ("Extracted features dimension: {}".format(features.shape))
        return features


    def GMM(self, x):
        """

        """
        model = GaussianMixture(n_components=self.max_clusters)
        model.fit(x)
        return model


    def Agglomerative(self, x):
        """

        """
        model = AgglomerativeClustering(n_clusters = self.max_clusters)
        model.fit(x)
        return model


    def kmeans(self, x):
        """
        """
        model = KMeans(n_clusters = self.max_clusters)
        model.fit(x)
        return model


    def birch(self, x, threshold = 0.01):
        """

        """
        model = Birch(threshold = threshold, n_clusters = self.max_clusters)
        model.fit(x)
        return model

    
    def dbscan(self, x, threshold = 0.01, min_samples = 0.01):
        """

        """
        model = DBSCAN(eps=threshold, min_samples = max(10, int(min_samples*len(x))))
        model.fit(x)
        return model


    def optics(self, x, threshold = 0.01, min_samples = 0.01):
        """

        """
        model = OPTICS(eps=threshold, min_samples = max(10, int(min_samples*len(x))))
        model.fit(x)
        return model

    
    def get_clusters(self, save_path=None):
        """

        """
        
        self.model = None
        if self.method == "kmeans":
            self.model = self.kmeans(self.features)
        elif self.method == 'agglomerative':
            self.model = self.Agglomerative(self.features)
        elif self.method == 'gmm':
            self.model = self.GMM(self.features)
        elif self.method == 'birch':
            self.model = self.birch(self.features)
        elif self.method == 'dbscan':
            self.model = self.dbscan(self.features)
        elif self.method == 'optics':
            self.model = self.optics(self.features)
        
        try:
            self.labels = self.model.predict(self.features)
        except:
            self.labels = self.model.fit_predict(self.features)

        if save_path:
            os.makedirs(save_path, exist_ok = True)
            with open(os.path.join(save_path, self.layer_name + '_clusters.cpickle'), 'wb') as file:
                pickle.dump({'layer_name': self.layer_name,
                                        'clusters': self.labels}, file)
            
        return self.labels 


    def plot_features(self, projection_dim = 2, label = None, save_path = None):
        """

        dim x: z x nfeatures
        projection_dim: feature rescale to .
        save_path: path to save plot
        """ 
        pca = PCA(n_components=projection_dim)
        pca.fit(self.features)
        X = pca.transform(self.features)
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()

        if projection_dim == 3:
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
            plt.cla()
            if not isinstance(label, np.ndarray): ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.nipy_spectral, edgecolor='k')
            else :
                for i in np.unique(label):
                    ax.scatter(X[label == i][:, 0],
                                X[label == i][:, 1],
                                X[label == i][:, 2], 
                                cmap=plt.cm.nipy_spectral, edgecolor='k')
        elif projection_dim == 2:
            if not isinstance(label, np.ndarray): plt.scatter(X[:,0], X[:,1], cmap=plt.cm.nipy_spectral, edgecolor='k')
            else:
                for i in np.unique(label):
                    plt.scatter(X[label == i][:,0],
                                X[label == i][:,1],
                                cmap=plt.cm.nipy_spectral, edgecolor='k')
        else:
            raise ValueError("Projection dimension > 3, cannot be ploted")

        if not save_path:
            plt.show()
        else:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'features_layer_{}_projection_{}.png'.format(self.layer_idx, projection_dim)))

    
    def plot_clusters(self, save_path = None):
        """
        dim x: z x nfeatures
        save_path: root path to save plot
        """
        try:
            label = self.labels
        except:
            raise ValueError("Model not generated yet, First run get_cluster, attr")

        self.plot_features(2, label, save_path)
        self.plot_features(3, label, save_path)
        return label


    def plot_weights(self, n = 3, save_path=None):
        """
        dim x: k x k x in_c x out_c
        """
        shape = self.weights.shape
        wt_idx = np.random.randint(0, shape[-1], n)
        rws = int(shape[-2]**0.5)
        cls = rws

        if not rws**2 == shape[-2]:
            rws = rws + 1
        
        feature = np.zeros((shape[0]*rws, shape[1]*cls))
        for ii in wt_idx:
            wt = self.weights[:,:,:, ii]
            for i in range(rws):
                for j in range(cls):
                    try:
                        feature[i*shape[0]: (i + 1)*shape[0], 
                            j*shape[1]: (j + 1)*shape[1]] = wt[:, :, j*rws + i]
                    except:
                        pass

            plt.clf()
            plt.imshow(feature)
            if not save_path:
                plt.show()
            else:
                os.makedirs(save_path, exist_ok = True)
                plt.savefig(os.path.join(save_path, 'wt_sample_layer_{}_idx_{}.png'.format(self.layer_idx, ii)))
