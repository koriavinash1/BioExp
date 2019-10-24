import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np

layer = 'conv2d_19'
class_ = 'class_1'
layers_to_consider = os.listdir('../results/Ablation/unet_flair/')
class_to_consider  = ['class_1', 'class_2', 'class_3']

for layer in layers_to_consider:
	for class_ in class_to_consider:
		abstudy = '../results/Ablation/unet_flair/{}/{}.csv'.format(layer, class_)
		dstudy  = '../results/Dissection/unet_flair/csv/{}.csv'.format(layer)

		a = pd.read_csv(abstudy)
		a = a.sort_values(['filter'], ascending=True)
		a_val = a['value'].values
		a_val = (a_val - np.min(a_val))/(np.max(a_val) - np.min(a_val))

		d = pd.read_csv(dstudy)
		d_val = d_val[class_].values
		d_val = (d_val - np.min(d_val))/(np.max(d_val) - np.min(d_val))

		corr = np.corrcoef(a_val, b_val)
		print ("{}: {} corr: {}".format(layer, class_, corr[0,1]))
