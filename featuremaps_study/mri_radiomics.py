import numpy as np
import pandas as pd
import pickle 
import os
from glob import glob
import nibabel as nib
import sys
sys.path.append('..')
from BioExp.helpers import utils, radfeatures


seq = 't2'
root_path = '../sample_vol/'
all_patients = glob(root_path+'*/*_' + seq + '.nii.gz')
all_masks    = glob(root_path+'*/*_seg.nii.gz')
assert len(all_patients) == len(all_masks) 

nclasses = 1infoclasses = {}
for i in range(1): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3)


for class_ in infoclasses.keys():
	save_path = './RadiomicAnalysis/unet_{}/MRI/{}/'.format(seq, class_)
	os.makedirs(save_path, exist_ok=True)

	for i, (vol_, mask_) in enumerate(zip(all_patients, all_masks)):

		vol  = nib.load(vol_).get_data()
		vol = (vol - np.mean(vol))/(np.std(vol))
		vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
		seg = nib.load(mask_).get_data()
		
		for _class_ in infoclasses[class_]:
			mask += seg == _class_

		mask = np.uint8(mask)
		pth = os.path.join(save_path, str(i))	
		os.makedirs(pth, exist_ok=True)
		feat_extractor = radfeatures.ExtractRadiomicFeatures(vol, mask, save_path = pth)
		df = feat_extractor.all_features()

