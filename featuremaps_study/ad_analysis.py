import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

layer = 'conv2d_19'
layers_to_consider = os.listdir('../results/Ablation/unet_flair/')
class_to_consider  = ['class_1', 'class_2', 'class_3']

for layer in layers_to_consider:
	for class_ in class_to_consider:
		try:
			abstudy = '../results/Ablation/unet_flair/{}/{}.csv'.format(layer, class_)
			dstudy  = '/media/brats/mirlproject2/parth/results_scaled/Dissection/unet_flair/csv/{}_dice_scores.csv'.format(layer)

			a = pd.read_csv(abstudy)
			a = a.sort_values(['filter'], ascending=True)
			a_val = a['value'].values
			a_val = 1. - ((a_val - np.min(a_val))/(np.max(a_val) - np.min(a_val)))

			d = pd.read_csv(dstudy)
			d_val = d[class_].values
			d_val = (d_val - np.min(d_val))/(np.max(d_val) - np.min(d_val))
			plt.clf()
			plt.plot (a_val, d_val)
			plt.xlabel("Ablation")
			plt.ylabel("Dissection")
			plt.savefig(layer+'.png')
			corr = np.corrcoef(a_val, d_val)
			print ("{}: {} corr: {}".format(layer, class_, corr[0,1]))
		except: pass

