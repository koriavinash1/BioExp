import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import vis
import cv2
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_cams

from ..helpers.utils import *

threshold = 0.5
num_images = 3
num_classes = 3

st = time()
def get_grad_dice(test_image, gt, model_type, save = True):
    global layer_dice
    counter = 0
    for layer in range(1, len(model.layers)):
        if 'conv' in model.layers[layer].name:
	
            if save:
                plt.figure(figsize=(30, 10))
                gs = gridspec.GridSpec(1, 3)
                gs.update(wspace=0.025, hspace=0.05)

            for class_ in range(1,4):
                grads_ = visualize_cam(dummy_model, layer_idx, filter_indices=class_, penultimate_layer_idx = layer,  
                                seed_input=test_image[None, ...], backprop_modifier='guided') 

                if save:
                    ax = plt.subplot(gs[class_ -1])
                    im = ax.imshow(grads_, cmap=plt.cm.RdBu)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    ax.tick_params(bottom='off', top='off', labelbottom='off' )
                    if class_ == 4:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.2)
                        cb = plt.colorbar(im, ax=ax, cax=cax )


                thresh_image = grads_ > threshold
                gt_mask = gt == class_
                score = (np.sum(thresh_image*gt_mask) + 1.0)*2.0/(np.sum(gt_mask*1. + thresh_image*1.) + 1.0)
                layer_dice[counter][class_ -1] += score
            counter += 1

            if save:
                plt.savefig(os.path.join(model_type, "gradientflow_"+model.layers[layer].name.replace('/', '_')+'.png'), bbox_inches='tight')



for model_type in ['dense']:
    model, weights_path, pb_path = load_seg_model(model_type)
    print (weights_path, pb_path)

    if not os.path.exists(model_type):
        os.mkdir(model_type)

    model.load_weights(weights_path)

    layer_idx = -1
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    
    num_layers = sum([model.layers[layer].name.__contains__('conv') for layer in range(1, len(model.layers))])
    
    layer_dice = np.zeros((num_layers, num_classes))
    for i in range(num_images):
        test_image, gt = load_vol(test_path[i], model_type, slice_ = 78)
        save = True if i == 0 else False 
        get_grad_dice(test_image, gt, model_type, save)
        print (i, time() - st)

    layer_dice = layer_dice/(num_images*1.0)
    np.save(model_type+'_layer_wise_dice.npy', layer_dice)
