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
from scipy.stats import chi2_contingency

from pgm.helpers.common import Node
from pgm.representation.LinkedListBN import Graph

class CausalGraph():
    """
        class to generate causal 
    """
    def __init__(self, model, weights_pth, classinfo=None):
        
        """
            model       : keras model architecture (keras.models.Model)
            weights_pth : saved weights path (str)
            layer_name  : name of the layer which needs to be ablated
            test_img    : test image used for ablation
        """     

        self.model      = model
        self.weights    = weights_pth
        self.classinfo  = classinfo
        self.noutputs   = len(self.model.outputs)

            
    def get_layer_idx(self, layer_name):
        """
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return idx
       

    def calc_MI(self, X,Y,bins):

       c_XY = np.histogram2d(X,Y,bins)[0]
       c_X = np.histogram(X,bins)[0]
       c_Y = np.histogram(Y,bins)[0]

       H_X = self.shan_entropy(c_X)
       H_Y = self.shan_entropy(c_Y)
       H_XY = self.shan_entropy(c_XY)

       MI = H_X + H_Y - H_XY
       return MI

    def shan_entropy(self, c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized* np.log2(c_normalized))  
        return H
        

    def MI(self, distA, distB, bins=100, random=100):
        """
        """
        x = distA.reshape(distA.shape[0], -1)
        y = distB.reshape(distB.shape[0], -1)

        idxs = np.arange(x.shape[-1])
        if random:
            idxs = np.random.choice(idxs, random)

        mi = []
        for i in idxs:
            mi.append(self.calc_MI(x[:, i], y[:,i], bins))
        return np.mean(mi)
        

    def get_link(self, nodeA_info, nodeB_info, dataset_path, loader, max_samples = 1):
        """
            get link between two nodes, nodeA, nodeB
            occlude at nodeA and observe changes in nodeB
            nodeA_info    : {'layer_name', 'feature_map_idxs'}
            nodeB_info    : {'layer_name', 'feature_map_idxs'}
        """

        nodeA_idx   = self.get_layer_idx(nodeA_info['layer_name'])
        nodeA_idxs  = nodeA_info['feature_map_idxs']
        nodeB_idx   = self.get_layer_idx(nodeB_info['layer_name'])
        nodeB_idxs  = nodeB_info['feature_map_idxs']

        total_filters = np.arange(np.array(self.model.layers[nodeA_idx].get_weights())[0].shape[-1])

        #########################

        modelT = Model(inputs = self.model.input, 
                    outputs = self.model.get_layer(nodeB_info['layer_name']).output)
        modelT.load_weights(self.weights, by_name=True)

        test_filters  = np.delete(total_filters, nodeA_idxs)
        layer_weights = np.array(self.model.layers[nodeA_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
                occluded_weights[0][:,:,:,j] = 0
                try: occluded_weights[1][j] = 0
                except: pass
        modelT.layers[nodeA_idx].set_weights(occluded_weights)


        #########################

        modelP = Model(inputs = self.model.input, 
                        outputs=self.model.get_layer(nodeB_info['layer_name']).output)
        modelP.load_weights(self.weights, by_name=True)
        modelP.layers[nodeA_idx].set_weights(occluded_weights)

        total_filters = np.arange(np.array(self.model.layers[nodeB_idx].get_weights())[0].shape[-1])
        test_filters  = np.delete(total_filters, nodeB_idxs)
        layer_weights = np.array(self.model.layers[nodeB_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
                occluded_weights[0][:,:,:,j] = 0
                try: occluded_weights[1][j] = 0
                except: pass
        modelP.layers[nodeB_idx].set_weights(occluded_weights)

        #########################
        true_distributionB = []
        predicted_distributionB = []

        input_paths = os.listdir(dataset_path)
        for i in range(len(input_paths) if len(input_paths) < max_samples else max_samples):
            input_, label_ = loader(os.path.join(dataset_path, input_paths[i]))
            true_distributionB.append(np.squeeze(modelT.predict(input_[None, ...])))
            predicted_distributionB.append(np.squeeze(modelP.predict(input_[None, ...])))

        return self.MI(np.array(true_distributionB), np.array(predicted_distributionB))


    def generate_graph(self, graph_info, dataset_path, dataloader, 
                            edge_threshold = None, save_path = None, 
                            verbose = False, max_samples=10):
        """
            graph_info: list of json: [{'layer_name',
                                        'filter_idxs',
                                        'concept_name',
                                        'description'}]
        """

        layers= []
        filter_idxs =[]
        concept_names=[]
        descp =[]
        for cinfo in graph_info:
            layers.append(cinfo['layer_name'])
            filter_idxs.append(cinfo['filter_idxs'])
            concept_names.append(cinfo['concept_name'])
            descp.append(cinfo['description'])
        
        layers=np.array(layers); filter_idxs=np.array(filter_idxs);
        concept_names=np.array(concept_names); descp=np.array(descp);
                
        if not edge_threshold:
            raise ValueError("Assign proper edge threshold")

        layer_names  = np.unique(layers)
        layer_names  = layer_names[np.argsort([int(idx.split('_')[-1]) for idx in layer_names])]

        node_ordering = []
        node_indexing = []

        for i in range(len(layer_names)):
            node_ordering.extend(concept_names[layers == layer_names[i]])
            node_indexing.extend([i]*sum(layers == layer_names[i]))
                
        node_indexing = np.array(node_indexing)
        node_ordering = np.array(node_ordering)
        
        rootNode = Node('Input')
        rootNode.info = {'concept_name': 'Input Image',
                            'layer_name': 'Placeholder',
                            'filter_idxs': [0],
                            'description': 'Input Image to a network'}
        self.causal_BN = Graph(rootNode)

        for ii, (idxi, nodei) in enumerate(zip(node_indexing, node_ordering)):
            nodei_info = {'concept_name': nodei, 
                            'layer_name': layers[concept_names == nodei][0], 
                            'filter_idxs': filter_idxs[concept_names == nodei][0],
                            'description': descp[concept_names == nodei][0]}

            try:
                Nodei = self.causal_BN.get_node(nodei)
                self.causal_BN.current_node.info = nodei_info
                Aexists = True
            except:
                Aexists = False

            if nodei_info['layer_name'] == layer_names[0]:
                self.causal_BN.add_node(nodei, parentNodes = ['Input'])
                self.causal_BN.get_node(nodei)
                self.causal_BN.current_node.info = nodei_info
                Aexists = True


            for jj, (idxj, nodej) in enumerate(zip(node_indexing[node_indexing > idxi], 
                                                        node_ordering[node_indexing > idxi])):

                nodej_info = {'concept_name': nodej, 
                                'layer_name': layers[concept_names == nodej][0], 
                                'filter_idxs': filter_idxs[concept_names == nodej][0],
                                'description': descp[concept_names == nodej][0]}

                link_info = self.get_link(nodei_info,
                                            nodej_info,
                                            dataset_path = dataset_path,
                                            loader = dataloader,
                                            max_samples = max_samples)

                try:
                    Nodej =self.causal_BN.get_node(nodej)
                    Bexists = True
                    self.causal_BN.current_node.info = nodej_info
                except:
                    Bexists = False
                

                print("[INFO: BioExp Graphs] Causal Relation between: {}, {}; edge weights: {}".format(nodei, 
                                        nodej, link_info))
                
                if link_info > edge_threshold:
                    if Aexists:
                        if not Bexists:
                            self.causal_BN.add_node(nodej,
                                        parentNodes = [nodei])
                            self.causal_BN.get_node(nodej)
                            self.causal_BN.current_node.info = nodej_info
                        else:
                            self.causal_BN.add_edge(nodei, nodej)
                    else:
                        pass


            self.causal_BN.print(rootNode)

        os.makedirs(save_path, exist_ok=True)
        pickle.dump({'graph': self.causal_BN, 'rootNode': rootNode}, 
                        open(os.path.join(save_path, 'causal_graph.pickle'), 'wb'))
        print("[INFO: BioExp Graphs] Causal Graph Generated")
        pass

    def perform_intervention(self):
        """
        find all active trails conditioned on output
        """
        pass