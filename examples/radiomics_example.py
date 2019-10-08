import numpy as np
import pandas as pd
import pickle 
import os
from glob import glob
import nibabel as nib
from radiomic_features import ExtractRadiomicFeatures
import cv2

nclasses = 4
layer = 17
_filter = 30
seq = 'flair'


save_path = '../results/RadiomicAnalysis/unet_{}/'.format(seq)
os.makedirs(save_path, exist_ok=True)
print('../results/unet_{}/conv2d_{}_{}.png'.format(seq, layer, _filter))
image = cv2.imread('../results/unet_{}/lucid/conv2d_{}_{}.png'.format(seq, layer, _filter), cv2.IMREAD_GRAYSCALE)
pth = os.path.join(save_path, str(layer), str(_filter))
mask = np.ones(image.shape)
os.makedirs(pth, exist_ok=True)
feat_extractor = ExtractRadiomicFeatures(image, mask, save_path = pth)
df = feat_extractor.all_features()