import numpy as np
import os
import sys
sys.path.append('../..')
from BioExp.spatial import flow
from BioExp.helpers import utils
from BioExp.helpers import losses
import SimpleITK as sitk
from tensorflow.keras.models import load_model
import keras
import tensorflow as tf
from glob import glob
from tensorflow.initializers import glorot_uniform as GlorotUniform
from keras.utils import CustomObjectScope

data_root_path = '../sample_vol/histopath'

model_path     = '/media/balaji/CamelyonProject/avinash/histopath_models/dense-net-sample-model.h5'
weights_path   = '/media/balaji/CamelyonProject/avinash/histopath_models/densenet-model-wts.14-0.09.h5'
save_path      = '/media/balaji/CamelyonProject/avinash/BioExpResults/'


imgs = glob(os.path.join(data_root_path, 'imgs/*'))
gts  = glob(os.path.join(data_root_path, 'masks/*'))

nclasses = 2

dices = []



model = load_model(model_path, custom_objects={'softmax_dice_loss': losses.softmax_dice_loss})
model.load_weights(weights_path)
print(model.summary())
for img, gt in zip(imgs, gts):
	print (img, gt)
	img = utils.load_images(img, mask=False)
	gt  = utils.load_images(gt, mask=True)
	print (img.shape, gt.shape)
	dice = flow.cam(model, img, gt, 
				nclasses = nclasses, 
				save_path = save_path, 
				layer_idx = -1, 
				threshold = 0.5,
				modifier = 'guided')
	print ("[BioExp:INFO Mean Layer Dice:] ", dice) 
	dices.append(dice)

dice = np.mean(dices, axis= 0)
print ("[BioExp:INFO Mean Layer Dice:] ", dice) 
