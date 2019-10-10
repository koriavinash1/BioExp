import numpy as np
import pandas as pd
import pickle 
import os
from glob import glob
import nibabel as nib
from radiomic_features import ExtractRadiomicFeatures

seq = 'flair'
root_path = '../sample_vol/'
all_patients = glob(root_path+'*/*_' + seq + '.nii.gz')
all_masks    = glob(root_path+'*/*_seg.nii.gz')
assert len(all_patients) == len(all_masks) 

nclasses = 4

# We need to preprocess the MRI images same as the model before getting radiomic features

for ii in range(nclasses):
	save_path = '../results/RadiomicAnalysis/unet_{}/MRI/class_{}/'.format(seq, ii)
	os.makedirs(save_path, exist_ok=True)

	for i, (vol_, mask_) in enumerate(zip(all_patients, all_masks)):
		if i >= 1: break

		vol  = nib.load(vol_).get_data()
		vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
		mask = nib.load(mask_).get_data()
		mask = np.uint8(mask == ii)
		pth = os.path.join(save_path, str(i))	
		os.makedirs(pth, exist_ok=True)
		feat_extractor = ExtractRadiomicFeatures(vol, mask, save_path = pth)
		df = feat_extractor.all_features()

