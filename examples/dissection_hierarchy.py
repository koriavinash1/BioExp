import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

model = 'densenet'
csv_root = 'results_ET/Dissection/' + model +'/csv'
layers = os.listdir(csv_root)

classes_to_consider = ['ET', 'CT',  'whole']
dice_thresh = 0.06

class_layer_count = {}
for class_ in classes_to_consider:
	class_layer_count[class_] = []

class_layer_count['layers'] = []
for layer in layers:
	df = pd.read_csv(os.path.join(csv_root, layer))
	class_layer_count['layers'].append(int(layer.split('.')[0].split('_')[1]))
	for class_ in classes_to_consider:
		class_layer_count[class_].append(np.sum(df[class_].values > dice_thresh)*1.0/len(df[class_].values) )



idx = np.argsort(class_layer_count['layers'])

ind = np.arange(len(class_layer_count['layers']))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(ind - width, np.array(class_layer_count['ET'])[idx], width,
			label='ET', color='r')
rects2 = ax.bar(ind , np.array(class_layer_count['CT'])[idx], width,
			label='CT', color='g')
rects4 = ax.bar(ind + 2*width, np.array(class_layer_count['whole'])[idx], width,
			label='whole', color='b')

ax.set_ylabel('class specific filter count')
ax.set_xticks(ind)
ax.set_xticklabels(np.array(class_layer_count['layers'])[idx])
ax.legend()
plt.savefig(model + '_feature_count.png')

