import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


layers_to_consider = os.listdir('/media/balaji/CamelyonProject/avinash/BioExp/featuremaps_study/ablation/results/Ablation/unet_flair/csv')
class_to_consider  = ['class_0', 'whole']

for layer in layers_to_consider:
	layer = layer.split('.')[0]
	for class_ in class_to_consider:
		#try:
			abstudy = '/media/balaji/CamelyonProject/avinash/BioExp/featuremaps_study/ablation/results/Ablation/unet_flair/csv/{}.csv'.format(layer)
			dstudy  = '/media/balaji/CamelyonProject/avinash/BioExp/featuremaps_study/dissection/results_scaled/Dissection/unet_flair/csv/{}_dice_scores.csv'.format(layer)

			a = pd.read_csv(abstudy)
			length = len(a)
			size = int(0.5*len(a))
			a = a.sort_values([class_], ascending=False)['feature'].values[:size]
	

			d = pd.read_csv(dstudy)
			d = d.sort_values([class_], ascending=False)['feature'].values[:size]
			
			similarity = 0
			for a_ in a:
				if a_ in d: similarity += 1

			print ("{}: {} similarity: {}, length: {}, size: {}, simila: {}".format(layer, class_, similarity*1.0/size, length, size, similarity))
		#except: pass

