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



def get_gram(test_image):
    channels = test_image.shape[-1]
    test_vec = test_image.reshape(-1, channels)
    test_vec = (test_vec - test_vec.min())/(test_vec.max() - test_vec.min())
    print (test_vec.min(), test_vec.max())	
    return np.matmul(test_vec.T, test_vec)/(1.*len(test_vec))


test_image, gt = load_vol(test_path[0], model_type='Uresnet', slice_ = 78)
image_shape = test_image.shape
channels = image_shape[-1]

gram_template = np.zeros((channels, channels), dtype='float16')

counter = 0
num_images = 2
from tqdm import tqdm

for ii in tqdm(range(num_images)):
    for slice_ in range(76, 80):
        try:
            test_image, gt = load_vol(test_path[ii], model_type='Uresnet', slice_ = slice_)
            gram_template += get_gram(test_image.astype('float16'))
            print (gram_template)
            counter += 1
        except : pass
np.save('GramTemplate.npy', gram_template/ (counter *1.0))
