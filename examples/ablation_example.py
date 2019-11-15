import keras
import numpy as np
import sys
sys.path.append('..')
from BioExp import spatial
from BioExp.helpers import utils
import SimpleITK as sitk
from BioExp.spatial import Ablation
#from BioExp.helpers.losses import *
from BioExp.helpers.losses import *
from BioExp.helpers.metrics import *
import pickle
from lucid.modelzoo.vision_base import Model
from BioExp.concept.feature import Feature_Visualizer
from tqdm import tqdm
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

seq = 'flair'

data_root_path = '../sample_vol/'

model_path = '../trained_models/U_resnet/ResUnet.h5'
weights_path = '../trained_models/U_resnet/ResUnet.40_0.559.hdf5'


model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
                                        'dice_whole_metric':dice_whole_metric,
                                        'dice_core_metric':dice_core_metric,
                                        'dice_en_metric':dice_en_metric})
model.load_weights(weights_path)

infoclasses = {}
for i in range(1): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3)


data_root_path = '../sample_vol/'
layer_name = 'conv2d_3'
test_image, gt = utils.load_vol_brats('../sample_vol/Brats18_CBICA_ARZ_1', slicen=78)
A = spatial.Ablation(model = moedl, 
				weights_pth = weights_path, 
				metric      = dice_label_coef, 
				layer_name  = layer_name, 
				test_image  = test_image, 
				gt 	    = gt, 
				classes     = infoclasses, 
				nclasses    = 4)

df = A.ablate_filter(step = 1)
