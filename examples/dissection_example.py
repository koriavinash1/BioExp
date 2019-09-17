import sys
sys.path.append('..')
from BioExp import spatial
from BioExp.helpers import utils
import SimpleITK as sitk
from keras.models import load_model
from losses import *


data_root_path = '../sample_vol/'

model_path = '../trained_models/U_resnet/ResUnet.h5'
weights_path = '../trained_models/U_resnet/ResUnet.40_0.559.hdf5'


model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
                                        'dice_whole_metric':dice_whole_metric,
                                        'dice_core_metric':dice_core_metric,
                                        'dice_en_metric':dice_en_metric})
model.load_weights(weights_path)

layer_name = 'conv2d_3'
dissector = spatial.Dissector(model=model,
                        layer_name = layer_name)

threshold_maps = dissector.get_threshold_maps(dataset_path = data_root_path,
                                                save_path  = '../results/Dissection/densenet/threshold_maps/',
                                                percentile = 85)


image, gt = utils.load_vol_brats('../sample_vol/Brats18_CBICA_ARZ_1', slicen=78)

maks_path = '../sample_vol/Brats18_CBICA_ARZ_1/mask.nii.gz'
ROI = sitk.GetArrayFromImage(sitk.ReadImage(maks_path))[78, :, :]
dissector.apply_threshold(image, threshold_maps, 
                        nfeatures=9, 
                        save_path='../results/Dissection/densenet/feature_maps/', 
                        ROI = ROI)

dissector.quantify_gt_features(image, gt, 
                        threshold_maps, 
                        nclasses=4, 
                        nfeatures=9, 
                        save_path='../results/Dissection/densenet/csv/',
                        save_fmaps=False, 
                        ROI = ROI)
