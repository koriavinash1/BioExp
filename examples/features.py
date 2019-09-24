import sys
sys.path.append('..')
from BioExp import spatial
from BioExp.helpers import utils
import SimpleITK as sitk
from keras.models import load_model
from losses import *

from lucid.modelzoo.vision_base import Model
import sys
sys.path.append('..')
from BioExp.concept.feature import Feature_Visualizer


model_path = '../trained_models/U_resnet/resnet.pb'
data_root_path = '../sample_vol/'
model_path = '../trained_models/U_resnet/ResUnet.h5'
weights_path = '../trained_models/U_resnet/ResUnet.40_0.559.hdf5'

model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
                                        'dice_whole_metric':dice_whole_metric,
                                        'dice_core_metric':dice_core_metric,
                                        'dice_en_metric':dice_en_metric})
model.load_weights(weights_path)


feature_maps = []
layers = []
classes = []

layers_to_consider = ['conv2d_3', 'conv2d_5', 'conv2d_7', 'conv2d_9', 'conv2d_11', 'conv2d_13', 'conv2d_17', 'conv2d_19', 'conv2d_21', 'conv2d_23']


# layer_name = 'conv2d_3'

for layer_name in layers_to_consider:
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

    _, df = dissector.quantify_gt_features(image, gt, 
                            threshold_maps, 
                            nclasses=4, 
                            nfeatures=9, 
                            save_path='../results/Dissection/densenet/csv/',
                            save_fmaps=False, 
                            ROI = ROI)

    # list all featuremap dice greater than 0.1
    dice_matrix = df.values[:, 1:]
    dice_matrix = dice_matrix > 0.1
    feature_info, class_info = np.where(dice_matrix)
    layers.extend([layer_name]*len(feature_info))
    feature_maps.extend(feature_info)
    classes.extend(class_info)

import ipdb
ipdb.set_trace()

# Initialize a class which loads a Lucid Model Instance with the required parameters

class Load_Model(Model):
  model_path = model_path
  image_shape = [None, 4, 240, 240]
  image_value_range = (0, 1)
  input_name = 'input_1'


# Initialize a Visualizer Instance
E = Feature_Visualizer(Load_Model, savepath = '../results/')

texture_maps = {}
for class_ in np.unique(classes):
    texture_maps['class'+str(class_)] = []


for layer_, feature_, class_ in zip(feature_maps, layers, classes):
    # Run the Visualizer
    texture_maps['class'+str(class_)].append(E.run(str(layer_) + '_' + str(feature_), channel = class_)) 

