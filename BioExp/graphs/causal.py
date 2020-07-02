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
        given layer names returns the 
        index corresponding to it
        """
        if layer_name == 'output':
            return len(self.model.layers) -1

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
        
        distA:  <ndarray>; tensor of any order, first axis should be sample axis (N, ...)
        distB:  <ndarray>; same dimensionality as distA
        bins :  <int>; bins used in creating histograms
        random: <float>; to seep up computation by randomly selecting n vectors 
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
    


    def get_classlink(self, node_info,
                        dataset_path,
                        loader,
                        max_samples=-1):
        r"""
        get link information between two nodes, nodeA, nodeB
        observation based on interventions
        
        This describes the existance of directed link between 
        nodeA -> nodeB not the other way

            
        node_info  : <json>; {'layer_name', 'filter_idxs'}
        dataset_path: <str> root director of dataset
        loader      : <function>; custom loader which takes image path
                        and return both image and corresponding path
                        simultaniously
        max_samples : <int>; maximum number of samples required for expectation
                         if -1 considers all images in provided root dir
        """
        

        node_idx   = self._get_layer_idx_(node_info['layer_name'])
        node_idxs  = node_info['filter_idxs']
        total_filters = np.arange(np.array(self.model.layers[node_idx].get_weights())[0].shape[-1])

        #-------------------------------------
        # Create duplicate models for estimating pre and post 
        # interventional distributions

        model = clone_model(self.model)

        #-------------------------------------
        # Occluding layer p weights

        test_filters  = np.delete(total_filters, node_idxs)
        layer_weights = np.array(model.layers[node_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
            occluded_weights[0][...,j] = 0
            try: occluded_weights[1][j] = 0
            except: pass
        model.layers[node_idx].set_weights(occluded_weights)

        #--------------------------------------
        # Distribution Estimation

        distribution = []

        input_paths = os.listdir(dataset_path)
        max_samples = len(input_paths) if max_samples == -1 else max_samples
        
        for i in range(len(input_paths) if len(input_paths) < max_samples else max_samples):
            input_, label_ = loader(os.path.join(dataset_path, input_paths[i]))
            intervened = np.squeeze(model.predict(input_[None, ...]))
            distribution.append(intervened)


        del model

        # numpification
        distribution = np.array(distribution)

        return np.mean(distribution, 
                        axis=tuple(np.delete(np.arange(len(distribution.shape)), 
                                        [1])))


    def get_link(self, nodeA_info, 
                 nodeB_info, 
                 dataset_path, 
                 loader,
                 max_samples = -1):
        r"""
        get link information between two nodes, nodeA, nodeB
        observation based on interventions
        
        This describes the existance of directed link between 
        nodeA -> nodeB not the other way

        $$NMI(\mathbb{Q}(\Phi(x ~|~ do(C_{-i}^p = 0), do(C_{-j}^q = 0))), \mathbb{P}(\Phi(x ~|~ do(C_{-i}^p = 0)))) > T$$
            
        nodeA_info  : <json>; {'layer_name', 'filter_idxs'}
        nodeB_info  : <json>; {'layer_name', 'filter_idxs'}
        dataset_path: <str> root director of dataset
        loader      : <function>; custom loader which takes image path
                        and return both image and corresponding path
                        simultaniously
        max_samples : <int>; maximum number of samples required for expectation
                         if -1 considers all images in provided root dir
        """
        

        nodeA_idx   = self._get_layer_idx_(nodeA_info['layer_name'])
        nodeA_idxs  = nodeA_info['filter_idxs']
        nodeB_idx   = self._get_layer_idx_(nodeB_info['layer_name'])
        nodeB_idxs  = nodeB_info['filter_idxs']
        
        total_filters = np.arange(np.array(self.model.layers[nodeA_idx].get_weights())[0].shape[-1])

        #-------------------------------------
        # Create duplicate models for estimating pre and post 
        # interventional distributions

        model_pre = clone_model(self.model)
        model_post = clone_model(self.model)


        model_pre = Model(inputs = model_pre.input, 
                        outputs = model_pre.get_layer(nodeB_info['layer_name']).output)
        model_post = Model(inputs = model_post.input, 
                        outputs = model_post.get_layer(nodeB_info['layer_name']).output)

        #-------------------------------------
        # Occluding layer p weights

        test_filters  = np.delete(total_filters, nodeA_idxs)
        layer_weights = np.array(model_pre.layers[nodeA_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
            occluded_weights[0][...,j] = 0
            try: occluded_weights[1][j] = 0
            except: pass
        model_pre.layers[nodeA_idx].set_weights(occluded_weights)
        model_post.layers[nodeA_idx].set_weights(occluded_weights)


        #--------------------------------------
        # Occluding layer q weights

        total_filters = np.arange(np.array(self.model.layers[nodeB_idx].get_weights())[0].shape[-1])
        test_filters  = np.delete(total_filters, nodeB_idxs)
        layer_weights = np.array(model_post.layers[nodeB_idx].get_weights().copy())
        occluded_weights = layer_weights.copy()
        for j in test_filters:
            occluded_weights[0][...,j] = 0
            try: occluded_weights[1][j] = 0
            except: pass
        model_post.layers[nodeB_idx].set_weights(occluded_weights)


        #--------------------------------------
        # Distribution Estimation

        pre_distribution = []
        post_distribution = []

        input_paths = os.listdir(dataset_path)
        max_samples = len(input_paths) if max_samples == -1 else max_samples
        
        for i in range(len(input_paths) if len(input_paths) < max_samples else max_samples):
            input_, label_ = loader(os.path.join(dataset_path, input_paths[i]))
            pre_intervened = np.squeeze(model_pre.predict(input_[None, ...]))
            post_intervened = np.squeeze(model_post.predict(input_[None, ...]))
            pre_distribution.append(pre_intervened)
            post_distribution.append(post_intervened)


        del model_post, model_pre

        # numpification
        pre_distribution = np.array(pre_distribution)
        post_distribution = np.array(post_distribution)

        return self.MI(pre_distribution, post_distribution)



    def generate_graph(self, graph_info, 
                           dataset_path, 
                           dataloader, 
                           nclasses = 2,
                           type     = 'segmentation',
                           edge_threshold = 0.5, 
                           save_path = None, 
                           verbose = True, 
                           max_samples=10):
        r"""
        
        Constructs entire concept graph based on the information provided
        
        
        
        graph_info: <list[json]>; [{'layer_name',
                                        'filter_idxs',
                                        'concept_name',
                                        'description'}]
        dataset_path: <str>; root director of dataset
        dataloader: <function>; custom loader which takes image path
                        and return both image and corresponding path
                        simultaniously
        nclasses: <int>; number of classes pixel wise or data wise
        type: <str>; one among ['segmentation', 'classification'] 
        edge_threshold: <float>; threshold value to form an edge
        save_path: <str>; path to folder to save all the files and generated graph in .pickle
                    format
        verbose: <bool>; provide log statements
        max_samples: <int>; maximum number of samples required for expectation
                         if -1 considers all images in provided root dir
        
        """
        if not type.lower() in ['segmentation', 'classification']: 
            raise NotImplementedError("[INFO: BioExp Graphs] allowed types are ['segmentation', 'classification']")

        # -------------------------------
        # seperating lists from json

        layers= []
        filter_idxs =[]
        concept_names=[]
        descp = []
        node_indexing = []
        for cinfo in graph_info:
            layers.append(cinfo['layer_name'])
            filter_idxs.append(cinfo['filter_idxs'])
            concept_names.append(cinfo['concept_name'])
            descp.append(cinfo['description'])
            node_indexing.append(cinfo['node_order'])

        layers = np.array(layers); filter_idxs = np.array(filter_idxs);
        concept_names = np.array(concept_names); descp = np.array(descp);
        node_indexing = np.array(node_indexing)    

        if not edge_threshold:
            raise ValueError("Assign proper edge threshold")


        node_ordering = []
        for i in np.sort(np.unique(node_indexing)):
            node_ordering.extend(concept_names[node_indexing == i])
                
        node_ordering = np.array(node_ordering)

        # -----------------------------------

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

            if idxi == 0:
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


        #--------------------------------
        # final class nodes
        print ("[INFO: BioExp Graphs] leaf node addition]")
        
        for nodei in self.causal_BN.get_leafnodes():
           
            link_info = self.get_classlink(nodei.info,
                                    dataset_path = dataset_path,
                                    loader = dataloader,
                                    max_samples = max_samples)

            if type.lower() == 'segmentation':
                class_ = np.argmax(link_info[1:]) + 1# removing background class
            elif type.lower() == 'classification':
                class_ = np.argmax(link_info)
            else:
                raise NotImplementedError("[INFO: BioExp Graphs] allowed types are ['segmentation', 'classification']")


            for cls_ in np.arange(nclasses)[link_info == link_info[class_]]:

                nodej = 'class' + str(cls_)
                nodej_info = {'concept_name': nodej, 
                            'layer_name': 'output', 
                            'filter_idxs': [],
                            'description': 'Output node Class_{}'.format(cls_)}
                
                try:
                    nodej = self.causal_BN.get_node(nodej)
                    Bexists = True
                    self.causal_BN.current_node.info = nodej_info
                except:
                    Bexists = False

                if not Bexists:
                    self.causal_BN.add_node(nodej,
                                parentNodes = [nodei.name])
                    self.causal_BN.get_node(nodej)
                    self.causal_BN.current_node.info = nodej_info
                else:
                    self.causal_BN.add_edge(nodei, nodej)

            if verbose: self.causal_BN.print(rootNode)

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