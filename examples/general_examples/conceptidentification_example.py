import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')
from BioExp.clusters.concept import ConceptIdentification
from BioExp.graphs import concept
from BioExp.helpers import utils
import SimpleITK as sitk
from keras.models import load_model
from BioExp.helpers.losses import *

from keras.backend.tensorflow_backend import set_session
import argparse

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session())

	
parser = argparse.ArgumentParser(description='feature study')
parser.add_argument('--seq', default='t1c', type=str, help='mri sequence')
parser = parser.parse_args()


seq_map = {'flair': 0, 't1': 1, 't2': 3, 't1c':2}
seq = parser.seq

print (seq)
model_path        = '/home/brats/parth/saved_models/model_{}/model-archi.h5'.format(seq)
weights_path      = '/home/brats/parth/saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)
data_root_path = '/home/pi/Projects/test-data/HGG/'


model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
                                        'dice_whole_metric':dice_whole_metric,
                                        'dice_core_metric':dice_core_metric,
                                        'dice_en_metric':dice_en_metric})
model.load_weights(weights_path)


def dataloader(nslice = 78):
	def loader(img_path, mask_path):
		image, gt =  utils.load_vol_brats(img_path, slicen=nslice)
		return image[:, :, seq_map[seq]][:,:, None], gt
	return loader

infoclasses = {}
# for i in range(4): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3)
infoclasses['ET'] = (3,)
infoclasses['CT'] = (1,3)
metric = dice_label_coef
layer_names = ['conv2d_2'] #, 'conv2d_5', 'conv2d_7','conv2d_9', 'conv2d_11', 'conv2d_13', 'conv2d_15', 'conv2d_17', 'conv2d_19', 'conv2d_21']

image, gt = utils.load_vol_brats('../../sample_vol/brats/Brats18_CBICA_AOP_1', slicen=105)
image = image[:, :, seq_map[seq]][:,:, None]
maks_path = '../../sample_vol/brats/Brats18_CBICA_AOP_1/mask.nii.gz'
ROI = sitk.GetArrayFromImage(sitk.ReadImage(maks_path))[105, :, :]

identifier = ConceptIdentification(model, weights_path, metric)
G = concept.ConceptGraph(model, weights_path, metric, layer_names)
clusters_info = G.get_concepts('.')

for i in range(len(clusters_info['concept_name'])):
	concept_info = {'concept_name': clusters_info['concept_name'][i], 'layer_name': clusters_info['layer_name'][i], 'filter_idxs': clusters_info['feature_map_idxs'][i]}
	identifier.check_robustness(concept_info, 
		                    save_path = 'cluster_robustness_results', 
		                    test_img = image,
		                    test_gt = gt,
				    save_all = True)

