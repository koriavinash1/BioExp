import pandas as pd
import numpy as np
from glob import glob
import os
seq = 'flair'
nclasses = 4

mriclass = []
amapclass = []

for class_ in range(nclasses):
	mri_features = glob('../results/RadiomicAnalysis/unet_{}/MRI/class_{}/*/firstorder_features.csv'.format(seq, class_))
	amaps_features = glob('../results/RadiomicAnalysis/unet_{}/amaps/class_{}/*/firstorder_features.csv'.format(seq, class_))

	mri_values = pd.read_csv(mri_features[0]).values
	amap_values = pd.read_csv(amaps_features[0]).values

	for m in mri_features[1:]:
		mri_values = np.concatenate([mri_values, pd.read_csv(m).values], axis=0)

	for am in amaps_features[1:]:
		amap_values = np.concatenate([amap_values, pd.read_csv(am).values], axis=0)

	print (mri_values.shape, amap_values.shape)
	mriclass.append(np.squeeze(np.mean(mri_values, axis=0)))
	amapclass.append(np.squeeze(np.mean(amap_values, axis=0)))

mriclass = np.array(mriclass)
amapclass = np.array(amapclass)

corr_matrix = np.zeros((nclasses, nclasses))

dist = lambda x,y : np.sum(x*y) # /((np.sum(x**2)**0.5) * (np.sum(y**2)**0.5))

for i in range(nclasses):
	for j in range(nclasses):
		corr_matrix[i, j] = dist(mriclass[i], amapclass[j])

print (corr_matrix)


