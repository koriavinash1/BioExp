from glob import glob
import sys
sys.path.append('..')
from BioExp.uncertainty import uncertainty
from BioExp.helpers.utils import load_vol_brats
from BioExp.helpers.losses import *

path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

test_path= path_HGG
np.random.seed(2022)
np.random.shuffle(test_path)
model = load_model('/home/pi/Projects/BioExp/trained_models/densedrop/densedrop.h5', 
                custom_objects={'gen_dice_loss':gen_dice_loss,
                                'dice_whole_metric':dice_whole_metric,
                                'dice_core_metric':dice_core_metric,
                                'dice_en_metric':dice_en_metric})

model.load_weights('/home/pi/Projects/BioExp/trained_models/densedrop/model_lrsch.hdf5', by_name = True)

if __name__ == '__main__':
    list_ = []
    for volume in [32, 20, 24, 53, 12, 14]:
        for slice_ in range(20, 140, 5):
            test_image, gt = load_vol_brats(test_path[volume], slice_, pad = 0)

            D = uncertainty(test_image)
            
            # for aleatoric
            mean, var = D.aleatoric(model, iterations = 50)
           
            # for epistemic
            mean, var = D.epistemic(model, iterations = 50)
 
            # for combined
            mean, var = D.combined(model, iterations = 50)
      
            print (np.mean(var, axis=(0,1,2)))
            list_.append(np.mean(var, axis=(0,1,2)))


    list_ = np.mean(list_, axis=0)
    print (list_)
