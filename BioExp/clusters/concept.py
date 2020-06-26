import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pdb
import os
import cv2 
import keras
import random
import numpy as np
from glob import glob
import SimpleITK as sitk
import pandas as pd
from ..helpers.utils import *
from ..spatial.dissection import Dissector
from ..spatial.flow import singlelayercam
from keras.models import Model
from skimage.transform import resize as imresize
from keras.utils import np_utils
from keras import layers
from keras.models import Sequential
import keras.backend as tf

import matplotlib.gridspec as gridspec
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure


class ConceptIdentification():
    """
        Network Dissection analysis

        model      : keras model initialized with trained weights
        layer_name : intermediate layer name which needs to be analysed
    """

    def __init__(self, model, weights_pth, metric, nclasses=4):

        self.model       = model
        self.metric      = metric
        self.weights     = weights_pth
        self.nclasses    = nclasses
        self.model.load_weights(self.weights, by_name = True)


    def _get_layer_idx(self, layer_name):
        """
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return idx

    def save_concepts(self, img, concepts, nrows, ncols, name, save_path=None):
        """
            creats a grid of image and saves if path is given

            img : test image
            concepts: all features vectors
            nrows : number of rows in an image
            ncols : number of columns in an image
            save_path : path to save an image
        """

        plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(wspace=0.025, hspace=0.05)
        
        for i in range(nrows):
            for j in range(ncols):
                try:
                    concept = concepts[:,:,i*nrows +(j)]

                    concept = np.ma.masked_where(concept == 0, concept)
                    ax = plt.subplot(gs[i, j])
                    im = ax.imshow(np.squeeze(img), cmap='gray')
                    im = ax.imshow(concept, alpha=0.5, cmap = plt.cm.RdBu, vmin = 0, vmax = 3)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    ax.tick_params(bottom='off', top='off', labelbottom='off' )
                except:
                    pass
        
        if save_path:
            if not os.path.exists(save_path): 
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, name+'.png'), bbox_inches='tight')
        else:
            plt.show()

    def identify(self, concept_info, 
                            dataset_path, 
                            save_path, 
                            loader,
                            test_img,
                            img_ROI = None):
        """
            test significance of each concepts

            concept: {'concept_name', layer_name', 'filter_idxs'}
            dataset_path: 
            save_path:
            loader:
            test_imgs:
            img_ROI:
        """
        layer_name = concept_info['layer_name']        
        self.dissector  = Dissector(self.model, layer_name)

        threshold_maps  = self.dissector.get_threshold_maps(dataset_path, save_path, percentile = 85, loader=loader)

        concepts = self.dissector.apply_threshold(test_img, threshold_maps,
                                            post_process_threshold = 80,
                                            ROI=img_ROI)

        node_idxs = concept_info['filter_idxs']
        concepts = concepts[:, :, node_idxs]

        if save_path:
            nrows = int(len(node_idxs)**.5) + 1
            self.save_concepts(test_img, concepts, nrows, nrows, concept_info['concept_name'], save_path = save_path)

        # some statistics on concepts
        mean_concept = np.round(np.mean(concepts, axis=2)[:,:,None])
        self.save_concepts(test_img, mean_concept, 1, 1, concept_info['concept_name']+'mean', save_path = save_path)

        return concepts

    def get_layer_idx(self, layer_name):
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return idx

    def flow_based_identifier(self, concept_info, 
                            save_path, 
                            test_img):
        """
            test significance of each concepts

            concept: {'concept_name', layer_name', 'filter_idxs'}
            dataset_path: 
            save_path:
            loader:
            test_imgs:
        """
        layer_name = concept_info['layer_name']        
        node_idxs = concept_info['filter_idxs']

        self.model.load_weights(self.weights, by_name = True)
        node_idx  = self.get_layer_idx(concept_info['layer_name'])
        total_filters = np.arange(np.array(self.model.layers[node_idx].get_weights())[0].shape[-1])
        # test_filters  = np.delete(total_filters, node_idxs)

        # layer_weights = np.array(self.model.layers[node_idx].get_weights().copy())
        # occluded_weights = layer_weights.copy()
        # for j in test_filters:
        #     occluded_weights[0][:,:,:,j] = 0
        #     try:
        #         occluded_weights[1][j] = 0
        #     except: pass
		
        """
        for j in node_idxs:
            occluded_weights[0][:,:,:,j] = 1.
            occluded_weights[1][j] = 1.
        """

        # self.model.layers[node_idx].set_weights(occluded_weights)
        model = Model(inputs = self.model.input, outputs=self.model.get_layer(concept_info['layer_name']).output)
  
        newmodel = Sequential()
        newmodel.add(model)
        newmodel.add(layers.Conv2D(1,1))
        
        # for ii in range(len(self.model.layers)):
        #    newmodel.layers[ii].set_weights(self.model.layers[ii].get_weights())
        newmodel.layers[-1].set_weights((np.ones((1, 1, len(total_filters), 1)), np.ones(1)))

        grad = singlelayercam(newmodel, test_img, 
                        nclasses = 1, 
                        save_path = save_path, 
                        name  = concept_info['concept_name'], 
                        st_layer_idx = -1, 
                        end_layer_idx = 1,
                        threshold = 0.5)
        print ("[INFO: BioExp Concept Identification] Identified Concept {} in layer {}".format(concept_info['concept_name'], layer_name))

        del model, newmodel
        return grad[0]


    def _gaussian_sampler_(self, data, size, ax=-1):
        shape = np.mean(data, ax).shape + (size,)
        return lambda: np.std(data, -1)[..., None] * np.random.randn(*list(shape)) + np.mean(data, -1)[..., None] # np.random.normal(loc=np.mean(data, axis=ax), scale=np.std(data, axis=ax), size=size)


    def concept_distribution(self, concept_info):
        """
            concept_info: {'concept_name', 'layer_name', 'filter_idxs'}

            return: weight_sampler, bias_sampler
        """

        layer_name = concept_info['layer_name']        
        node_idxs = concept_info['filter_idxs']

        self.model.load_weights(self.weights, by_name = True)
        node_idx  = self.get_layer_idx(concept_info['layer_name'])

        layer_weights = np.array(self.model.layers[node_idx].get_weights().copy())
        concept_weights = layer_weights[0][:,:,:, node_idxs]
        try:
            concept_biases  = layer_weights[1][node_idxs]
            return (self._gaussian_sampler_(concept_weights, len(node_idxs)), self._gaussian_sampler(concept_biases, len(node_idxs)))
        except:
            return (self._gaussian_sampler_(concept_weights, len(node_idxs)))


    def concept_robustness(self, concept_info,
                            test_img,
                            nmontecarlo=3):
        """
            test significance of each concepts

            concept: {'concept_name', layer_name', 'filter_idxs'}
            dataset_path: 
            save_path:
            loader:
            test_imgs:
            img_ROI:
        """
        layer_name = concept_info['layer_name']        
        node_idxs = concept_info['filter_idxs']

        self.model.load_weights(self.weights, by_name = True)
        node_idx  = self.get_layer_idx(concept_info['layer_name'])
        total_filters = np.arange(np.array(self.model.layers[node_idx].get_weights())[0].shape[-1])
        test_filters  = np.delete(total_filters, node_idxs)

        layer_weights = np.array(self.model.layers[node_idx].get_weights().copy())
		
        occluded_weights = layer_weights.copy()

        for j in test_filters:
            occluded_weights[0][:,:,:,j] = 0
            try:
                occluded_weights[1][j] = 0
            except: pass

        weight_sampler = self._gaussian_sampler_(occluded_weights[0][:,:,:,node_idxs], len(node_idxs)) 

        try:
            bias_sampler = self._gaussian_sampler_(occluded_weights[1][:,:,:,node_idxs], len(node_idxs))
        except: pass
        
        gradlist = []
        for _ in range(nmontecarlo):

            occluded_weights[0][:,:,:,node_idxs] = weight_sampler()
            try:
                occluded_weights[1][node_idxs] = bias_sampler()
            except: pass
        
            self.model.layers[node_idx].set_weights(occluded_weights)
            model = Model(inputs = self.model.input, outputs=self.model.get_layer(concept_info['layer_name']).output)
  
            newmodel = Sequential()
            newmodel.add(model)
            newmodel.add(layers.Conv2D(1,1))
        
            # for ii in range(len(self.model.layers)):
            #    newmodel.layers[ii].set_weights(self.model.layers[ii].get_weights())
            newmodel.layers[-1].set_weights((np.ones((1, 1, len(total_filters), 1)), np.ones(1)))

            nclass_grad = singlelayercam(newmodel, test_img, 
                        nclasses = 1, 
                        name  = concept_info['concept_name'], 
                        st_layer_idx = -1, 
                        end_layer_idx = 1,
                        threshold = 0.5)
            gradlist.append(nclass_grad[0])
	
        try:
            del bias_sampler
        except: pass       
        del model, newmodel
        return np.array(gradlist)


    def check_robustness(self, concept_info,
                            save_path, 
                            test_img,
                            save_all = False,
                            nmontecarlo = 4):

        actual_grad = self.flow_based_identifier(concept_info,
                                               save_path = None,
                                               test_img = test_img)
        montecarlo_grad = self.concept_robustness(concept_info,
                                              test_img,
                                              nmontecarlo=nmontecarlo)

        if save_path:
            plt.clf()
            if save_all:
                plt.figure(figsize=(10*(nmontecarlo + 1), 10))
                gs = gridspec.GridSpec(1, nmontecarlo + 1)
                gs.update(wspace=0.025, hspace=0.05)

                ax = plt.subplot(gs[0])
                im = ax.imshow(np.squeeze(test_img), vmin=0, vmax=1)
                im = ax.imshow(actual_grad, cmap=plt.get_cmap('jet'), vmin=0, vmax=1, alpha=0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.set_title('actual')
                ax.tick_params(bottom='off', top='off', labelbottom='off' )
                
                for ii in range(nmontecarlo):
                    ax = plt.subplot(gs[ii + 1])
                    im = ax.imshow(np.squeeze(test_img), vmin=0, vmax=1)
                    im = ax.imshow(montecarlo_grad[ii], cmap=plt.get_cmap('jet'), vmin=0, vmax=1, alpha=0.5)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    ax.set_title('sampled')
                    ax.tick_params(bottom='off', top='off', labelbottom='off')
            else:
                plt.figure(figsize=(10*(2), 10))
                gs = gridspec.GridSpec(1, 2)
                gs.update(wspace=0.025, hspace=0.05)

                ax = plt.subplot(gs[0])
                im = ax.imshow(np.squeeze(test_img), vmin=0, vmax=1)
                im = ax.imshow(actual_grad, cmap=plt.get_cmap('jet'), vmin=0, vmax=1, alpha=0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.set_title('actual')
                ax.tick_params(bottom='off', top='off', labelbottom='off' )
                
                ax = plt.subplot(gs[1])
                im = ax.imshow(np.squeeze(test_img), vmin=0, vmax=1)
                im = ax.imshow(np.mean(montecarlo_grad, axis=0), cmap=plt.get_cmap('jet'), vmin=0, vmax=1, alpha=0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.set_title('actual')
                ax.tick_params(bottom='off', top='off', labelbottom='off' )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cb = plt.colorbar(im, ax=ax, cax=cax )
            os.makedirs(save_path, exist_ok = True)
            plt.savefig(os.path.join(save_path, concept_info['concept_name'] +'_robustness.png'), bbox_inches='tight')
            
        return np.mean(montecarlo_grad, axis=0)
