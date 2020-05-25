import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
import os
from keras import activations

from ..helpers.utils import *

import vis
from vis.visualization import visualize_cam
from vis.utils import utils

def singlelayercam(model, img,
        nclasses = 2,
        save_path = None,
        name = None,
        end_layer_idx = 3,
        st_layer_idx = -1,
        threshold = 0.5,
        modifier='guided'):
    """
    """
    # model.layers[-1].activation = activations.linear
    # model = utils.apply_modifications(model)
    # print(model.summary())
    
    layer_dice = np.zeros((1, nclasses))
    layer_info = np.zeros((1, nclasses))
    if save_path:
        plt.figure(figsize=(10*nclasses, 10))
        gs = gridspec.GridSpec(1, nclasses)
        gs.update(wspace=0.025, hspace=0.05)
    
    nclass_grad = []
    for i  in range(nclasses):
        grads_ = visualize_cam(model, st_layer_idx, filter_indices=i, penultimate_layer_idx = end_layer_idx,  
                    seed_input = img[None, ...], backprop_modifier = modifier)
        if save_path:
            ax = plt.subplot(gs[i])
            im = ax.imshow(np.squeeze(img), vmin=0, vmax=1)
            im = ax.imshow(grads_, cmap=plt.get_cmap('jet'), alpha=0.5, vmin=0, vmax=1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.tick_params(bottom='off', top='off', labelbottom='off' )

        nclass_grad.append(grads_)
        
    if save_path:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cb = plt.colorbar(im, ax=ax, cax=cax )
        os.makedirs(save_path, exist_ok = True)
        plt.savefig(os.path.join(save_path, name +'.png'), bbox_inches='tight')
    return np.array(nclass_grad)


def cam(model, img, gt, 
    nclasses = 2, 
    save_path = None, 
    layer_idx = -1, 
    threshold = 0.5,
        dice = True,
    modifier = 'guided'):
    """
    """
    model.layers[layer_idx].activation = activations.linear
    # model = utils.apply_modifications(model)
    # print(model.summary())
    num_layers = sum([model.layers[layer].name.__contains__('conv') for layer in range(1, len(model.layers))])
    
    layer_dice = np.zeros((num_layers, nclasses))
    layer_cams = []
    
    counter = 0
    for layer in range(1, len(model.layers)):
        if 'conv' not in model.layers[layer].name: continue
        nclass_cam = []
        if save_path:
            plt.figure(figsize=(30, 10))
            gs = gridspec.GridSpec(1, nclasses)
            gs.update(wspace=0.025, hspace=0.05)

        for class_ in range(nclasses):
            grads_ = visualize_cam(model, layer_idx, filter_indices=class_, penultimate_layer_idx = layer,  
                        seed_input = img[None, ...], backprop_modifier = modifier)
            nclass_cam.append(grads_)
            if save_path:
                ax = plt.subplot(gs[class_])
                im = ax.imshow(grads_, cmap='jet')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.tick_params(bottom='off', top='off', labelbottom='off' )
                if class_ == nclasses:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.2)
                    cb = plt.colorbar(im, ax=ax, cax=cax )

            if dice:
                thresh_image = grads_ > threshold
                gt_mask = gt == class_
                score = (np.sum(thresh_image*gt_mask))*2.0/(np.sum(gt_mask*1. + thresh_image*1.) + 1.e-3)
                layer_dice[counter][class_ -1] += score
        counter += 1

        if save_path:
            os.makedirs(save_path, exist_ok = True)
            plt.savefig(os.path.join(save_path, model.layers[layer].name.replace('/', '_')+'.png'), bbox_inches='tight')
            
        layer_cams.append(nclass_cam)
    
    if dice:
        return np.array(layer_cams), layer_dice
    
    return np.array(layer_cams)

    
