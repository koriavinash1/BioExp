import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from keras.models import Model
from skimage.transform import resize as imresize
from keras.utils import np_utils

import matplotlib.gridspec as gridspec
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure


class ConceptIdentification():
    """
        Network Dissection analysis

        model      : keras model initialized with trained weights
        layer_name : intermediate layer name which needs to be analysed
    """

    def __init__(self, model, weights_pth, metric):

        self.model       = model
        self.metric      = metric
        self.weights     = weights_pth
        self.model.load_weights(self.weights, by_name = True)


    def _get_layer_idx(self, layer_name):
        """
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return idx

    def identify(self, concept_info, 
                            dataset_path, 
                            save_path, 
                            loader,
                            test_imgs,
                            img_ROI = None):
        """
            test significance of each concepts

            concept: {'layer_name', 'filter_idxs'}
            dataset_path: 
            save_path:
            loader:
            test_imgs:
            img_ROI:
        """
        layer_name = concept_info['layer_name']        
        self.dissector  = Dissector(self.model, layer_name)
        threshold_maps  = self.dissector.get_threshold_maps(dataset_path, save_path, percentile = 85, loader=loader)

        concepts = self.dissector.apply_threshold(test_imgs, threshold_maps,
                                            save_path = save_path,
                                            nfeatures = 25,
                                            post_process_threshold = 80,
                                            ROI=img_ROI)

        node_idxs = concept_info['filter_idxs']
        concepts = concepts[:, :, node_idxs]
        # some statistics on concepts
        return concepts

        

    
