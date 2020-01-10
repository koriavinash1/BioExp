from lucid.modelzoo.vision_base import Model
import sys
sys.path.append('..')
from BioExp.concept.feature import Feature_Visualizer
import matplotlib.pyplot as plt
import os
import numpy as np

# Initialize a class which loads a Lucid Model Instance with the required parameters
class Load_Model(Model):

  model_path = '../../saved_models/U_resnet/parth_pc.pb'
  image_shape = [None, 1, 240, 240]
  image_value_range = (0, 10)
  input_name = 'input_1'

# print ("pre load ...........................")
# # Initialize a Visualizer Instance
# E = Feature_Visualizer(Load_Model, savepath = '../results/', regularizer_params={'L1':0, 'rotate':0, 'TV': 0, 'jitter':0})
# print ("loaded...................................")
# # Run the Visualizer
# a = E.run(layer = 'conv2d_23', class_ = 'None', channel = 50, transforms=True,  gram_coeff = 1e-3, 
# 	style_template = '/home/parth/Interpretable_ML/Brain-tumor-segmentation/template_image.npy')

fig=plt.figure()

tv_ = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
gram = [1, 10, 100, 200, 1000]

for i in range(5):
	# ax = fig.add_subplot(1, 2, i)
	E = Feature_Visualizer(Load_Model, savepath = '../results/', regularizer_params={'L1':1e-5, 'jitter':8, 'TV': tv_[i]})
	print ("loaded...................................")
	# Run the Visualizer
	a = E.run(layer = 'conv2d_21', class_ = 'None', channel = 45, transforms=True, gram_coeff = 1e-4, style_template = '/home/parth/Interpretable_ML/Brain-tumor-segmentation/template_image.npy')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
	plt.imshow(np.reshape(a[-1], (240, 240)), cmap='gray',
	               interpolation='bilinear', vmin=0., vmax=1.)

	plt.savefig('../results/hyperparameter_progression/TV/TV_fig4_flair_test_{}.png'.format(i), bbox_inches='tight')
# # plt.show()