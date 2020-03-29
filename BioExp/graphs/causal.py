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
from scipy.ndimage.morphology import (binary_dilation, 
                                    generate_binary_structure)

from ppgm.helpers.common import Node
from ppgm.representation.LinkedListBN import Graph

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
        """
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return idx
        
        
    def KLDivergence(self, distA, distB):
        """
        """
        kl = np.sum(np.where(distA != 0, distA * np.log(distA / distB), 0))
        return kl
        

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
                
        total_filters = np.arange(np.array(self.model.layers[nodeA_idx].get_weights())[0].shape[-1])
        test_filters  = np.delete(total_filters, nodeA_idxs)
        layer_weights = np.array(self.model.layers[nodeA_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
                occluded_weights[0][:,:,:,j] = 0
                try: occluded_weights[1][j] = 0
                except: pass
        self.model.layers[nodeA_idx].set_weights(occluded_weights)
        modelT = Model(inputs = self.model.input, 
                    outputs = self.model.get_layer(nodeB_info['layer_name']).output)


        total_filters = np.arange(np.array(self.model.layers[nodeB_idx].get_weights())[0].shape[-1])
        test_filters  = np.delete(total_filters, nodeB_idxs)
        layer_weights = np.array(self.model.layers[nodeB_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
                occluded_weights[0][:,:,:,j] = 0
                try: occluded_weights[1][j] = 0
                except: pass
        self.model.layers[nodeB_idx].set_weights(occluded_weights)
        modelP = Model(inputs = self.model.input, 
                        outputs=self.model.get_layer(nodeB_info['layer_name']).output)

        #########################
        true_distributionB = []
        predicted_distributionB = []

        input_paths = os.listdir(dataset_path)
        for i in range(len(input_paths) if len(input_paths) < max_samples else max_samples):
            input_, label_ = loader(os.path.join(dataset_path, input_paths[i]), 
                                os.path.join(dataset_path, 
                                input_paths[i]).replace('mask', 
                                                                    'label').replace('labels', 'masks'))
            true_distributionB.append(np.squeeze(modelT.predict(input_[None, ...])))
            predicted_distributionB.append(np.squeeze(modelP.predict(input_[None, ...])))


        return self.KLDivergence(np.array(true_distributionB), np.array(predicted_distributionB))


    def generate_graph(self, graph_info, dataset_path, dataloader, 
                            edge_threshold = None, save_path = None, 
                            verbose = False, max_samples=10):
        """
        """
        layers   = graph_info['layer_name']
        concept_names = graph_info['concept_name']
        filter_idxs   = graph_info['feature_map_idxs']
                
        if not edge_threshold:
            raise ValueError("Assign proper edge threshold")

        layer_names  = np.unique(layers)
        layer_names  = layer_names[np.argsort([int(idx.split('_')[-1]) for idx in layer_names])]

        node_ordering = []
        node_indexing = []

        for i in renage(len(layer_names)):
            node_ordering.extend(concept_names[layers == layer_names[i]])
            node_indexing.extend([i]*sum(layers == layer_names[i]))
                
        node_indexing = np.array(node_indexing)
        node_ordering = np.array(node_ordering)
        
        rootNode = Node('Input')
        self.causal_BN = Graph(rootNode)

        # self.causal_graph = np.zeros((len(node_ordering), len(node_ordering)))
        # info = {'idxi': [], 'idxj': [], 'nodei': [], 'nodej': []}

        for ii, (idxi, nodei) in enumerate(zip(node_indexing, node_ordering)):
            nodei_info = {'concept_name': nodei, 
                            'layer_name': layers[concept_names == nodei], 
                            'feature_map_idxs': filter_idxs[concept_names == nodei]}


            Nodei = Node(nodei)
            Nodei.info = nodei_info

            if nodei_info['layer_name'] == layer_names[0]:
                self.causal_BN.add_node(Nodei, parentNode = rootNode)


            for jj, (idxj, nodej) in enumerate(zip(node_indexing[node_indexing > idxi], 
                                                        node_ordering[node_indexing > idx])):

                nodej_info = {'concept_name': nodej, 
                                'layer_name': layers[concept_names == nodej], 
                                'feature_map_idxs': filter_idxs[concept_names == nodej]}

                Nodej = Node(nodej)
                Nodej.info = nodej_info
                
                link_info = self.get_link(nodei_info,
                                            nodej_info,
                                            dataset_path = dataset_path,
                                            loader = dataloader,
                                            max_samples = max_samples)
                
                if link_info > edge_threshold:
                        self.causal_BN.add_node(Nodej,
                                        parentNode = Nodei)
                        Nodei.Distribution.append(link_info)

                

                print("Causal Relation between: {}, {}; edge probability: {}".format(nodei, 
                                        nodej, link_info))


                # self.causal_graph[ii, jj] = link_info
                # info['nodei'].append(nodei_info)
                # info['nodej'].append(nodej_info)
                # info['idxi'].append(ii)
                # info['idxj'].append(jj)

        os.makedirs(save_path, exist_ok=True)
        # info['graph'] = self.causal_graph
        # pickle.dump(info, open(os.path.join(save_path, 'causal_graph_info.pickle'), 'wb'))
        pickle.dump({'graph': self.causal_BN, 'rootNode': rootNode}, 
                        open(os.path.join(save_path, 'causal_graph_info.pickle'), 'wb'))
        print("Causal Graph Generated")
        pass

    def perform_intervention(self):
        """
        find all active trails conditioned on output
        """
        pass
