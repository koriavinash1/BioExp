#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[5]:

"""
get_ipython().system('pip install SimpleITK')
get_ipython().system('pip install pillow')
get_ipython().system('pip install scipy==1.1.0')
get_ipython().system('pip install git+git://github.com/tensorflow/lucid.git --upgrade --no-deps')
get_ipython().system('pip install keras')
get_ipython().system('pip install scikit-image')
get_ipython().run_line_magic('cd', 'Brain-tumor-segmentation')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
"""

import numpy as np
import random
from glob import glob
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.gridspec as gridspec
# from evaluation_metrics import *

path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

test_path=path_HGG+path_LGG
np.random.seed(2022)
np.random.shuffle(test_path)


def normalize_scheme(slice_not):
    '''
        normalizes each slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
    '''
    normed_slices = np.zeros(( 4,155, 240, 240))
    for slice_ix in range(4):
        normed_slices[slice_ix] = slice_not[slice_ix]
        for mode_ix in range(155):
            normed_slices[slice_ix][mode_ix] = _normalize(slice_not[slice_ix][mode_ix])

    return normed_slices    


def _normalize(slice):

    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    image_nonzero = slice[np.nonzero(slice)]

    if np.std(slice)==0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp==tmp.min()]=-9
        return tmp

def load_vol(filepath_image, model_type, slice_):

    '''
    segment the input volume
    INPUT   (1) str 'filepath_image': filepath of the volume to predict 
            (2) bool 'show': True to ,
    OUTPUt  (1) np array of the predicted volume
            (2) np array of the corresping ground truth
    '''

    #read the volume
    flair = glob( filepath_image + '/*_flair.nii.gz')
    t2 = glob( filepath_image + '/*_t2.nii.gz')
    gt = glob( filepath_image + '/*_seg.nii.gz')
    t1s = glob( filepath_image + '/*_t1.nii.gz')
    t1c = glob( filepath_image + '/*_t1ce.nii.gz')
    
    t1=[scan for scan in t1s if scan not in t1c]
    if (len(flair)+len(t2)+len(gt)+len(t1)+len(t1c))<5:
        print("there is a problem here!!! the problem lies in this patient :")
    scans_test = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i])) for i in range(len(scans_test))]


    test_im=np.array(test_im).astype(np.float32)
    test_image = test_im[0:4]
    gt=test_im[-1]
    gt[gt==4]=3

    #normalize each slice following the same scheme used for training
    test_image = normalize_scheme(test_image)

    #transform teh data to channels_last keras format
    test_image = test_image.swapaxes(0,1)
    test_image=np.transpose(test_image,(0,2,3,1))
    
    test_image, gt = np.array(test_image[slice_]), np.array(gt[slice_])
    if model_type == 'dense':
        npad = ((8, 8), (8, 8), (0, 0))
        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=0)
        npad = ((8, 8), (8, 8))
        gt = np.pad(gt, pad_width=npad, mode='constant', constant_values=0)
    return test_image, gt




def predict_volume(model, test_image, gt):
    
    test_image = test_image[None, ...]
    gt = gt[None, ...]
    
    prediction = model.predict(test_image, batch_size=1) 
    prediction_unshaped = prediction.copy()
    prediction = np.argmax(prediction, axis=-1)
    prediction=prediction.astype(np.uint8)
    #reconstruct the initial target values .i.e. 0,1,2,4 for prediction and ground truth
    prediction[prediction==3]=4
    gt[gt==3]=4
    
    
    plt.subplot(1,2,1)
    plt.imshow(prediction[0])
    plt.subplot(1,2,2)
    plt.imshow(gt[0])
    plt.show()
    
    return np.array(prediction), np.array(prediction_unshaped), np.array(gt)


# ## Model definition and weights path
# 
# ### Model selection for analysis

# In[7]:


resnet_model_path = 'trained_models/U_resnet/ResUnet.h5'
resnet_weights_path = 'trained_models/U_resnet/ResUnet.15_0.491.hdf5'
resnet_pb_path = 'trained_models/U_resnet/resnet.pb'


sunet_model_path = 'trained_models/SimUnet/FCN.h5'
sunet_weights_path = 'trained_models/SimUnet/SimUnet.40_0.060.hdf5'
sunet_pb_path = 'trained_models/SimUnet/SUnet.pb'


dense_model_path = 'trained_models/densenet_121/densenet121.h5'
dense_weights_path = 'trained_models/densenet_121/densenet.55_0.522.hdf5'
dense_pb_path = 'trained_models/densenet_121/densenet.pb'


shallow_model_path = 'trained_models/shallowunet/shallow_unet.h5'
shallow_weights_path = 'trained_models/shallowunet/shallow_weights.hdf5'
shallow_pb_path = 'trained_models/shallowunet/shallow_unet.pb'


from keras.models import load_model
from models import *
from losses import *


def load_seg_model(model_='uresnet'):
    
#     model = unet_densenet121_imagenet((240, 240), weights='imagenet12')
#     model.load_weights(weights_path)
    
    if model_ == 'uresnet':
        model = load_model(resnet_model_path, custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(resnet_weights_path)
        return model, resnet_weights_path, resnet_pb_path
    
    elif model_ == 'fcn':
        model = load_model(sunet_model_path, custom_objects={'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(sunet_weights_path)
        return model, sunet_weights_path, sunet_pb_path
    
    elif model_  == 'dense':
        model = load_model(dense_model_path, custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(dense_weights_path)
        return model, dense_weights_path, dense_pb_path

    elif model_  == 'shallow':
        model = load_model(shallow_model_path, custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(shallow_weights_path)
        return model, shallow_weights_path, shallow_pb_path


# ## Load model and model summary

# In[8]:

import vis
import cv2
from vis.utils import utils
import matplotlib.pyplot as plt

from vis.visualization import visualize_cam
from vis.visualization import visualize_saliency, overlay
from keras import activations
import matplotlib.cm as cm
import keras
from keras import models as md
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    dummy_model = md.clone_model(model)
    dummy_model.load_weights(weights_path)

    layer_idx = -1
    dummy_model.layers[layer_idx].activation = activations.linear
    dummy_model = utils.apply_modifications(dummy_model)
    
    num_layers = sum([model.layers[layer].name.__contains__('conv') for layer in range(1, len(model.layers))])
    
    layer_dice = np.zeros((num_layers, num_classes))
    for i in range(num_images):
        test_image, gt = load_vol(test_path[i], model_type, slice_ = 78)
        save = True if i == 0 else False 
        get_grad_dice(test_image, gt, model_type, save)
        print (i, time() - st)

    layer_dice = layer_dice/(num_images*1.0)
    np.save(model_type+'_layer_wise_dice.npy', layer_dice)
    plt.clf()
    plt.figure(figsize=(10, 10))
    for i in range(1,4):
    	plt.plot(layer_dice[:, i-1], linewidth=3.0)
    plt.xlabel("Layer Index")
    plt.ylabel("Class Dice")
    plt.legend(['Class0', 'Class1', 'Class2'])
    plt.savefig(os.path.join(model_type, 'layer_dice.png'), bbox_inches='tight')

