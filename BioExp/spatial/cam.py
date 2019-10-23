import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency, overlay
from keras.utils import CustomObjectScope

def get_cam(model, img, layer_idx=-1, custom_objects=None, label = None):

	with CustomObjectScope(custom_objects):
	    for i, modifier in enumerate(['guided', 'relu']):
	        plt.figure()
	        print('Modifier: {}'.format(modifier))
	        plt.suptitle("vanilla" if modifier is None else modifier)
	            
	        # 20 is the imagenet index corresponding to `ouzel`
	        grads = visualize_cam(model, layer_idx, filter_indices=1, 
	                              seed_input=img, backprop_modifier=modifier)        
	        # Lets overlay the heatmap onto original image.    
	        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
	        print(jet_heatmap.shape)
	        plt.imshow(overlay(jet_heatmap, np.squeeze(img)))
	        plt.savefig('/home/brats/parth/dsi-capstone/results/{}_{}.png'.format(label, modifier))
	        plt.close()
	        print('Modifier {} Done'.format(i))
