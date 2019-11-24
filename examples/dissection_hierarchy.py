import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

model = 'densenet'
csv_root = 'results/Dissection/' + model +'/csv'
layers = os.listdir(csv_root)

classes_to_consider = ['class_1', 'class_2', 'class_3']
dice_thresh = 0.6

class_layer_count = {}
for class_ in classes_to_consider:
	class_layer_count[class_] = []

class_layer_count['layers'] = []
for layer in layers:
	df = pd.read_csv(os.path.join(csv_root, layer))
	class_layer_count['layers'].append(int(layer.split('.')[0].split('_')[1]))
	for class_ in classes_to_consider:
		class_layer_count[class_].append(np.sum(df[class_].values > dice_thresh))



idx = np.argsort(class_layer_count['layers'])

ind = np.arange(len(class_layer_count['layers']))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(ind - width, np.array(class_layer_count['class_1'])[idx], width,
			label='class_1', color='r')
rects2 = ax.bar(ind , np.array(class_layer_count['class_2'])[idx], width,
			label='class_2', color='g')
rects2 = ax.bar(ind + width, np.array(class_layer_count['class_3'])[idx], width,
			label='class_3', color='b')


ax.set_ylabel('class specific filter count')
ax.set_xticks(ind)
ax.set_xticklabels(np.array(class_layer_count['layers'])[idx])
ax.legend()
plt.savefig(model + '_feature_count.png')

