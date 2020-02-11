import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from BioExp import spatial
from BioExp.helpers import utils
import SimpleITK as sitk
from keras.models import load_model
from BioExp.helpers.losses import *
from keras.backend.tensorflow_backend import set_session
import argparse

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

	
parser = argparse.ArgumentParser(description='feature study')
parser.add_argument('--seq', default='flair', type=str, help='mri sequence')
parser = parser.parse_args()


seq_map = {'flair': 0, 't1': 1, 't2': 3, 't1c':2}
seq = parser.seq


print (seq)
model_path        = '../../saved_models/model_{}/model-archi.h5'.format(seq)
weights_path      = '../../saved_models/model_{}/model-wts-{}.hdf5'.format(seq, seq)


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

layer_names = ['conv2d_2', 'conv2d_3'] #, 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12', 'conv2d_13', 'conv2d_14', 'conv2d_15', 'conv2d_16', 'conv2d_17', 'conv2d_18', 'conv2d_19', 'conv2d_20', 'conv2d_21']

infoclasses = {}
# for i in range(4): infoclasses['class_'+str(i)] = (i,)
infoclasses['whole'] = (1,2,3)
infoclasses['ET'] = (3,)
infoclasses['CT'] = (1,3)

for layer_name in layer_names:
	try:
		dissector = spatial.Dissector(model=model,
			                layer_name = layer_name, seq='all')
	except:
		continue

	threshold_maps = dissector.get_threshold_maps(dataset_path = data_root_path,
		                                        save_path  = 'results_ET/Dissection/simnet/threshold_maps/',
		                                        percentile = 85,
							loader=dataloader())

	image, gt = utils.load_vol_brats('../sample_vol/brats/Brats18_CBICA_AOP_1', slicen=105)
	image = image[:, :, seq_map[seq]][:,:, None]
	maks_path = '../sample_vol/brats/Brats18_CBICA_AOP_1/mask.nii.gz'
	ROI = sitk.GetArrayFromImage(sitk.ReadImage(maks_path))[105, :, :]
	dissector.apply_threshold(image, threshold_maps, 
		                nfeatures=25, 
		                save_path='.', 
		                ROI = ROI)

	dissector.quantify_gt_features(image, gt, 
		                threshold_maps, 
		                nclasses=infoclasses, 
		                nfeatures=25, 
		                save_path='results_ET/Dissection/simnet/csv/',
		                save_fmaps=False, 
		                ROI = ROI)

