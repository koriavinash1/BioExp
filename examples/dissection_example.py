import sys
sys.path.append('..')
from BioExp import spatial
from BioExp.helpers import utils
import SimpleITK as sitk
from keras.models import load_model
from BioExp.helpers.losses import *


data_root_path = '/home/brats/parth/test-data/HGG/'

model_path = '../trained_models/U_resnet/ResUnet.h5'
weights_path = '../trained_models/U_resnet/ResUnet.40_0.559.hdf5'


model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
                                        'dice_whole_metric':dice_whole_metric,
                                        'dice_core_metric':dice_core_metric,
                                        'dice_en_metric':dice_en_metric})
model.load_weights(weights_path)


def dataloader(nslice = 78):
	def loader(img_path, mask_path):
		image, gt =  utils.load_vol_brats(img_path, slicen=nslice)
		return image, gt
	return loader

layer_names = ['conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12', 'conv2d_13', 'conv2d_14', 'conv2d_15', 'conv2d_16', 'conv2d_17', 'conv2d_18', 'conv2d_19', 'conv2d_20', 'conv2d_21']

for layer_name in layer_names:
	try:
		dissector = spatial.Dissector(model=model,
			                layer_name = layer_name, seq='all')
	except:
		continue

	threshold_maps = dissector.get_threshold_maps(dataset_path = data_root_path,
		                                        save_path  = 'results/Dissection/densenet/threshold_maps/',
		                                        percentile = 85,
							loader=dataloader())


	image, gt = utils.load_vol_brats('../sample_vol/Brats18_CBICA_AOP_1', slicen=105)

	maks_path = '../sample_vol/Brats18_CBICA_AOP_1/mask.nii.gz'
	ROI = sitk.GetArrayFromImage(sitk.ReadImage(maks_path))[105, :, :]
	dissector.apply_threshold(image, threshold_maps, 
		                nfeatures=25, 
		                save_path='results/Dissection/densenet/feature_maps/', 
		                ROI = ROI)

	dissector.quantify_gt_features(image, gt, 
		                threshold_maps, 
		                nclasses=4, 
		                nfeatures=None, 
		                save_path='results/Dissection/densenet/csv/',
		                save_fmaps=False, 
		                ROI = ROI)
