import numpy as np
import os
import sys
sys.path.append('..')
from BioExp.spatial import flow
from BioExp.helpers import utils
from BioExp.helpers import losses
import SimpleITK as sitk
from keras.models import load_model


data_root_path = '../sample_vol/histopath'

model_path     = '/media/balaji/CamelyonProject/avinash/histopath_models/dense-net-sample-model.h5'
weights_path   = '/media/balaji/CamelyonProject/avinash/histopath_models/densenet-model-wts.14-0.09.h5'
save_path      = '/media/balaji/CamelyonProject/avinash/BioExpResults/'

model = load_model(model_path, custom_objects={'softmax_dice_loss': losses.softmax_dice_loss})
model.load_weights(weights_path)

imgs = glob(os.path.join(data_root_path, 'imgs/*'))
gts  = glob(os.path.join(data_root_path, 'masks/*'))

nclasses = 2


for img, gt in zip(imgs, gts):
	img = utils.load_images(img)
	gt  = utils.load_images(gts)
	try: 
		dice += spatial.cam(model, img, gt, 
				nclasses = nclasses, 
				save_path = save_path, 
				layer_idx = -1, 
				threshol = 0.5,
				modifier = 'guided')
	except:
		dice = spatial.cam(model, img, gt, 
				nclasses = 2, 
				save_path = save_path, 
				layer_idx = -1, 
				threshol = 0.5,
				modifier = 'guided')	
	

dice = dice/(1.0*len(imgs))
print ("[BioExp:INFO Layer Wise Dice:] ", dice) 
