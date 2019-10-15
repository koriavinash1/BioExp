import numpy as np
import pandas as pd 

layers_to_consider = ['conv2d_3', 'conv2d_5', 'conv2d_7','conv2d_9','conv2d_17','conv2d_19','conv2d_21']

for layer in layers_to_consider:
	print(layer)
	for _class in range(1, 4):
		

		d_scores = pd.read_csv('/media/balaji/CamelyonProject/parth/BioExp/results/Dissection/unet_flair/csv/{}_dice_scores.csv'.format(layer))

		d_scores = d_scores['class_{}'.format(_class)]

		a_scores = pd.read_csv('/media/balaji/CamelyonProject/parth/BioExp/results/Ablation/unet_flair/{}/class_{}.csv'.format(layer, _class))

		a_scores = a_scores.sort_values(['filter'], ascending = True)
		#print(a_scores)

		a_scores = a_scores['value']

		print(np.corrcoef(a_scores, d_scores)[0][1])
