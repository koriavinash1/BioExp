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

from keras.models import Model, clone_model
from keras.utils import np_utils
from tqdm import tqdm
from skimage.transform import resize as imresize

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import (binary_dilation, 
                                    generate_binary_structure)
from scipy.stats import chi2_contingency

from pgm.helpers.common import Node
from pgm.representation.LinkedListBN import Graph

EPS = np.finfo(np.float32).eps

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
        self.model.load_weights(self.weights)

            
    def _get_layer_idx_(self, layer_name):
        r"""
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return idx
       

    def _calc_MI_(self, X,Y,bins):
        r"""
        calculates normalized MI:
        
        NMI = 2*I(X,Y)/(H(X) + H(Y))
        """


        c_XY = np.histogram2d(X,Y,bins)[0]
        c_X = np.histogram(X,bins)[0]
        c_Y = np.histogram(Y,bins)[0]

        H_X = self._shan_entropy_(c_X) 
        H_Y = self._shan_entropy_(c_Y) 
        H_XY = self._shan_entropy_(c_XY) 
        

        MI = H_X + H_Y - H_XY

        return 2.*MI/(H_X + H_Y)

    def _shan_entropy_(self, c):
        r"""
        calculates shanan's entropy
        
        H(X) = E(xlog_2(x))
        """
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized* np.log2(c_normalized))  
        return H + EPS
        

    def MI(self, distA, distB, bins=100, random=0.05):
        r"""
        calculates mutual information between two 
        given distribution
        
        distA: tensor of any order, first axis should be sample axis (N, ...)
        distB: same dimensionality as distA
        bins : bins used in creating histograms
        random: to seep up computation by randomly selecting n vectors 
                and considers expectation
                (% information between (0, 1])
        """
        
        assert distA.shape == distB.shape, \
                "Dimensionality mismatch between two distributions"
        
        
        x = distA.reshape(distA.shape[0], -1)
        y = distB.reshape(distB.shape[0], -1)

        idxs = np.arange(x.shape[-1])
        if random:
            idxs = np.random.choice(idxs, int(random*x.shape[-1]))

        mi = []
        for i in idxs:
            mi.append(self._calc_MI_(x[:, i], y[:,i], bins))
        return np.mean(mi)
        

    def get_link(self, nodeA_info, 
                 nodeB_info, 
                 dataset_path, 
                 loader,
                 max_samples = -1):
        r"""
        get link information between two nodes, nodeA, nodeB
        observation based on interventions
        
        𝑀𝐼(𝐶𝑞𝑗, 𝑑𝑜(𝐶𝑝𝑖=1) | 𝑑𝑜(𝐶𝑝−𝑖=0), 𝑑𝑜(𝐶𝑞−𝑗=0))>𝑇
            
        nodeA_info  : {'layer_name', 'filter_idxs'}
        nodeB_info  : {'layer_name', 'filter_idxs'}
        dataset_path: <str> root director of dataset
        loader      : custom loader which takes image path
                        and return both image and corresponding path
                        simultaniously
        max_samples : maximum number of samples required for expectation
                         if -1 considers all images in provided root dir
        """
        

        nodeA_idx   = self._get_layer_idx_(nodeA_info['layer_name'])
        nodeA_idxs  = nodeA_info['filter_idxs']
        nodeB_idx   = self._get_layer_idx_(nodeB_info['layer_name'])
        nodeB_idxs  = nodeB_info['filter_idxs']
        
        total_filters = np.arange(np.array(self.model.layers[nodeA_idx].get_weights())[0].shape[-1])

        modelT = clone_model(self.model)
        modelP = clone_model(self.model)

        modelT = Model(inputs = modelT.input, 
                        outputs = modelT.get_layer(nodeB_info['layer_name']).output)
        modelP = Model(inputs = modelP.input, 
                        outputs = modelP.get_layer(nodeB_info['layer_name']).output)

        #########################
        test_filters  = np.delete(total_filters, nodeA_idxs)
        layer_weights = np.array(modelT.layers[nodeA_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
                occluded_weights[0][:,:,:,j] = 0
                try: occluded_weights[1][j] = 0
                except: pass
        modelT.layers[nodeA_idx].set_weights(occluded_weights)
        modelP.layers[nodeA_idx].set_weights(occluded_weights)


        #########################
        total_filters = np.arange(np.array(self.model.layers[nodeB_idx].get_weights())[0].shape[-1])
        test_filters  = np.delete(total_filters, nodeB_idxs)
        layer_weights = np.array(modelP.layers[nodeB_idx].get_weights().copy())
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
        max_samples = len(input_paths) if max_samples == -1 else max_samples
        
        for i in range(len(input_paths) if len(input_paths) < max_samples else max_samples):
            input_, label_ = loader(os.path.join(dataset_path, input_paths[i]))
            pre_intervened = np.squeeze(modelT.predict(input_[None, ...]))
            post_intervened = np.squeeze(modelP.predict(input_[None, ...]))
            true_distributionB.append(pre_intervened)
            predicted_distributionB.append(post_intervened)


        del modelP, modelT
        return self.MI(np.array(true_distributionB), np.array(predicted_distributionB))


    def generate_graph(self, graph_info, 
                           dataset_path, 
                           dataloader, 
                           edge_threshold = 0.5, 
                           save_path = None, 
                           verbose = True, 
                           max_samples=10):
        r"""
        
        Constructs entire concept graph based on the information provided
        
        
        
        graph_info: list of json: [{'layer_name',
                                        'filter_idxs',
                                        'concept_name',
                                        'description'}]
        dataset_path: <str> root director of dataset
        dataloader  : custom loader which takes image path
                        and return both image and corresponding path
                        simultaniously
        edge_threshold: <float> threshold value to form an edge
        save_path : <str> path to folder to save all the files and generated graph in .pickle
                    format
        verbose: <bool> provide log statements
        max_samples : maximum number of samples required for expectation
                         if -1 considers all images in provided root dir
        
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
                
                if verbose:
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