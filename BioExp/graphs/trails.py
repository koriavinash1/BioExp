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
from ..clusters.concept import ConceptIdentification

from keras.models import Model

from pgm.helpers.trails import findTrails
from pgm.helpers.common import Node
from pgm.representation.LinkedListBN import Graph

class EstimateTrails(object):
    r"""
    """
    def __init__(self, model, weights_pth, graph, root_node, metric, classinfo=None):
        r"""
            model       : keras model architecture (keras.models.Model)
            weights_pth : saved weights path (str)
            metric      : metric to compare prediction with gt, for example dice, CE
            layer_name  : name of the layer which needs to be ablated
            test_img    : test image used for ablation
        """     

        self.model      = model
        self.weights    = weights_pth
        self.graph      = graph
        self.root_node  = root_node
        self.metric     = metric
        self.classinfo  = classinfo
        self.noutputs   = len(self.model.outputs)
        self.identifier = ConceptIdentification(self.model, self.weights, self.metric)
            
    def get_layer_idx(self, layer_name):
        r"""
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layer_name:
                return idx
            
    
    def trails(self, start_node, end_node, image=None, gt=None, visual=True, save_path=None):
        r"""
        """
        if visual:
            if (image.all() and gt.all()):
                raise ValueError("improper argument fot test_img or test_gt")
                
        ftrails = findTrails(self.root_node, start_node, end_node)
        trails = []
        trailsdescription = []
        visualtrails = []

        for trailidx, ntrail in enumerate(ftrails.trails):
            trail = ''
            traildescription = ''
            concept_imgs = []
            
            for node in ntrail:

                if node.name == end_node:
                    trail += '  ({})  '.format(node.name)
                    traildescription += '  ({})  '.format(node.info['description'])
                else:
                    trail += '  ({})  ->'.format(node.name)
                    traildescription += '  ({})  ->'.format(node.info['description'])


                if visual:
                    if node.info['concept_name'] == 'Input Image':
                        concept_imgs.append(image)
                    else:
                        concept_imgs.append(self.identifier.check_robustness(node.info, 
                                                       test_img = image,
                                                       save_path = None,
                                                       nmontecarlo = 1))
            trails.append(trail)
            trailsdescription.append(traildescription)
            visualtrails.append(concept_imgs)
            print ("[INFO: BioExp Trails]" + "="*5 + " New trail " + "="*5)
            print (trail)
            print (traildescription)

        if visual:
            for trailidx, concept_imgs in enumerate(visualtrails):
                for i, cimg in enumerate(concept_imgs):
                    plt.subplot(1, len(concept_imgs), i+1)
                    plt.imshow(np.squeeze(cimg), cmap='jet', vmin=0, vmax=1)
                    plt.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False, labelleft=False,labelbottom=False)
                
                    if save_path:
                        pth = os.path.join(save_path, 'SN_{}__EN_{}'.format(start_node, end_node))
                        os.makedirs(pth, exist_ok=True)
                        plt.savefig(os.path.join(pth, '{}.png'.format(trailidx)), bbox_inches='tight')
                    else:
                        plt.show()
                    
            return trails, trailsdescription, visualtrails
        return trails, trailsdescription