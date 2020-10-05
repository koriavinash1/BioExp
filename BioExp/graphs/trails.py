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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    def __init__(self, model, weights_pth, graph, root_node, metric=None, ntrails= 1, classinfo=None):
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
        self.ntrails    = ntrails
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
        
        ftrails = findTrails(self.root_node, start_node, end_node)
        trails = []
        trailsdescription = []
        visualtrails = []
        pth = os.path.join(save_path, 'SN_{}__EN_{}'.format(start_node, end_node))
        os.makedirs(pth, exist_ok=True)


        for trailidx, ntrail in enumerate(ftrails.trails):
            if trailidx >= self.ntrails: continue

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
                    elif not node.info['concept_name'].__contains__('class'):
                        concept_imgs.append(self.identifier.flow_based_identifier(node.info, 
                                                       test_img = image,
                                                       save_path = None))
                    else:
                        pass
            trails.append(trail)
            trailsdescription.append(traildescription)
            visualtrails.append(concept_imgs)
            print ("[INFO: BioExp Trails]" + "="*5 + " New trail " + "="*5)
            print (trail)
            print (traildescription)

            if visual:
                plt.figure(figsize=(5*(len(concept_imgs) + 1), 5))
                gs = gridspec.GridSpec(1, (len(concept_imgs) + 1))
                gs.update(wspace=0.025, hspace=0.05)

                for i, cimg in enumerate(concept_imgs):
                    ax = plt.subplot(gs[i])
                    orimage = np.squeeze(image)

                    if len(orimage.shape) == 3:
                        im = ax.imshow(orimage, vmin=0, vmax=1)
                    else:
                        im = ax.imshow(orimage, cmap='gray', vmin=0, vmax=1)
                    
                    if i > 0:
                        cimg = np.squeeze(cimg)
                        im = ax.imshow(cimg, cmap=get_transparent_cmap('Greens'), vmin=0, vmax=1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    ax.tick_params(bottom='off', top='off', labelbottom='off' )

                ax = plt.subplot(gs[i + 1])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.tick_params(bottom='off', top='off', labelbottom='off' )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.2)
                cb = plt.colorbar(im, ax=ax, cax=cax )


                if save_path:
                    os.makedirs(save_path, exist_ok = True)
                    plt.savefig(os.path.join(pth, '{}.png'.format(trailidx)), bbox_inches='tight')
                else:
                    plt.show()
        
        if visual:              
            return trails, trailsdescription, visualtrails
        return trails, trailsdescription