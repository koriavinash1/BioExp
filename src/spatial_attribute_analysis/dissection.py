from keras.models import load_model
from glob import glob
import keras
import numpy as np
from losses import *
import random
from keras.models import Model
from scipy.misc import imresize
from keras.utils import np_utils
import SimpleITK as sitk
import pdb
import matplotlib.pyplot as plt
import os
from scipy.ndimage.measurements import label
import cv2 
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import matplotlib.gridspec as gridspec

# from evaluation_metrics import *

path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

test_path=path_HGG+path_LGG
np.random.seed(2022)
np.random.shuffle(test_path)

model_list = []
filter_list = []
layer_list = []
image_list = []
c0_dice = []
c1_dice = []
c2_dice = []
c3_dice = []
c4_dice = []

def normalize_scheme(slice_not):
    '''
        normalizes each slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
    '''
    normed_slices = np.zeros(( 4,155, 240, 240))
    for slice_ix in range(4):
        normed_slices[slice_ix] = slice_not[slice_ix]
        for mode_ix in range(155):
            normed_slices[slice_ix][mode_ix] = _normalize(slice_not[slice_ix][mode_ix])

    return normed_slices    


def _normalize(slice):

    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    image_nonzero = slice[np.nonzero(slice)]

    if np.std(slice)==0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp==tmp.min()]=-9
        return tmp

def load_vol(filepath_image, model_type, slice_):

    '''
    segment the input volume
    INPUT   (1) str 'filepath_image': filepath of the volume to predict 
            (2) bool 'show': True to ,
    OUTPUt  (1) np array of the predicted volume
            (2) np array of the corresping ground truth
    '''

    #read the volume
    flair = glob( filepath_image + '/*_flair.nii.gz')
    t2 = glob( filepath_image + '/*_t2.nii.gz')
    gt = glob( filepath_image + '/*_seg.nii.gz')
    t1s = glob( filepath_image + '/*_t1.nii.gz')
    t1c = glob( filepath_image + '/*_t1ce.nii.gz')
    mask = glob( filepath_image + '/mask.nii.gz')
    
    t1=[scan for scan in t1s if scan not in t1c]
    if (len(flair)+len(t2)+len(gt)+len(t1)+len(t1c))<5:
        print("there is a problem here!!! the problem lies in this patient :")
    scans_test = [flair[0], t1[0], t1c[0], t2[0], gt[0], mask[0]]
    test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i])) for i in range(len(scans_test))]


    test_im=np.array(test_im).astype(np.float32)
    test_image = test_im[0:4]
    gt=test_im[-2]
    gt[gt==4]=3
    mask = test_im[-1]


    #normalize each slice following the same scheme used for training
    test_image = normalize_scheme(test_image)

    #transform teh data to channels_last keras format
    test_image = test_image.swapaxes(0,1)
    test_image=np.transpose(test_image,(0,2,3,1))
    print (mask.shape)
    test_image, gt, mask_ = np.array(test_image[slice_]), np.array(gt[slice_]), np.array(mask[slice_])
    if model_type == 'dense':
        npad = ((8, 8), (8, 8), (0, 0))
        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=0)
        npad = ((8, 8), (8, 8))
        gt = np.pad(gt, pad_width=npad, mode='constant', constant_values=0)
        mask_ = np.pad(mask_, pad_width=npad, mode='constant', constant_values=0)
    return test_image, gt, mask_


def perform_postprocessing(img, threshold=800):
    c,n = label(img)
    nums = np.array([np.sum(c==i) for i in range(1, n+1)])
    # print (nums)
    selected_components = np.array([threshold<num for num in nums])
    selected_components[np.argmax(nums)] = True
    mask = np.zeros_like(img)
    # print(selected_components.tolist())
    for i,select in enumerate(selected_components):
        if select:
            mask[c==(i+1)]=1
    return mask


import pdb
class Dissector():

    def __init__(self, model, weights, data_path, layer_name):
        model = load_model(model, custom_objects={'gen_dice_loss':gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(weights)

        self.model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        for i in range(0, len(self.model.layers)):
            self.model.layers[i].set_weights(model.layers[i].get_weights())
            self.model.layers[i].trainable = False

        self.path = glob(data_path)[:5]
        self.layer_name = layer_name

        pass

    def get_threshold_maps(self, model_type, percentile):

        fmaps = []
        for i in range(len(self.path)):
            for j in range(5, 145, 5):
                print (self.path[i])
                input_, label_, _ = load_vol(self.path[i], model_type, slice_=j)
                output = np.squeeze(self.model.predict(input_[None, ...]))
                fmaps.append(output)
        # pdb.set_trace()
        fmaps = np.array(fmaps)
        mean_maps = np.mean(fmaps, axis=0)
        # std_maps = np.std(fmaps, axis=0)
        # threshold_maps = mean_maps + 2.*std_maps
        threshold_maps = np.percentile(fmaps, percentile, axis=0)

        # np.save('ModelDissection_layer_{}.npy'.format(self.layer_name), threshold_maps)
        np.save('ModelDissection_layer_fmaps_{}.npy'.format(self.layer_name), fmaps)
        return threshold_maps


    def apply_threshold(self, test_image, gt, mask_, threshold_maps, image_name, model_name, layer_name):

        fmaps = np.squeeze(self.model.predict(test_image[None, ...]))
        masks = fmaps >= threshold_maps
        masks = 1.*(masks)

        shape = test_image.shape[:-1]
        resized_masks = np.zeros((shape[0], shape[1], masks.shape[2]))
        kernel = np.ones((2, 2), np.uint8) 

        class_filters = np.zeros(3)

        for i in range(11, 12):
           
            resized_img = imresize(masks[:,:,i], shape, interp='nearest')
            post_processed_img = perform_postprocessing(resized_img)
            eroded_img = (cv2.dilate(post_processed_img, kernel, iterations=1))/255
            eroded_img = eroded_img*mask_
            dice = []
            for class_ in range(1,4):
                mask = gt == class_
                class_dice = (np.sum(mask*(eroded_img>0)) + 1e-5)*2.0/(np.sum(mask*1.) + np.sum((eroded_img*1.0) > 0) + 1e-5) 
                dice.append(class_dice)
                if class_dice > 4e-3:
                    class_filters[class_ -1] += 1
    
            model_list.append(model_name)
            filter_list.append(i)
            layer_list.append(layer_name)
            image_list.append(image_name)
            c0_dice.append(dice[0])
            c1_dice.append(dice[1])
            c2_dice.append(dice[2])
            c3_dice.append(np.sum(((1.*(gt>0))*eroded_img > 0) + 1e-5)*2.0/(np.sum(1.*(gt>0)) + np.sum((eroded_img*1.0 >0)) + 1e-5))
            c4_dice.append(np.sum(((1.*np.logical_or(gt==1,gt==3))*eroded_img > 0) + 1e-5)*2.0/(np.sum(1.*np.logical_or(gt==1,gt==3)) + np.sum((eroded_img*1.0) > 0) + 1e-5))
                

            resized_masks[:,:,i] = eroded_img

        channels = threshold_maps.shape[2]
        rows = 6 # int(channels**0.5)
        cols = 6
        """
        plt.figure(figsize=(100, 100))
        gs = gridspec.GridSpec(rows, cols)
        gs.update(wspace=0.025, hspace=0.05)
        
        for i in range(rows):
            for j in range(cols):
                concept = resized_masks[:,:,i*rows +(j+1)]
                concept = np.ma.masked_where(concept == 0, concept)

                ax = plt.subplot(gs[i, j])
                im = ax.imshow(test_image[:,:,3], cmap='gray')
                im = ax.imshow(concept, alpha=0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                ax.tick_params(bottom='off', top='off', labelbottom='off' )
                # plt.subplot(7, 7, i*7 +(j+1))
                # plt.imshow(resized_masks[:,:,i*7 +(j+1)], cmap='gray')


        plt.savefig(image_name +'_'+ model_name + '_' +layer_name+'_.png', bbox_inches='tight')
        # plt.show()
        """
        return class_filters

if __name__ == "__main__":
    import pandas as pd
    layer_wise_filters = []
    models_to_consider = ['dense']
    # layers_to_consider = ['conv2d_3', 'conv2d_5', 'conv2d_13', 'conv2d_17',  'conv2d_21']
    layers_to_consider = ['conv2d_10'] # for densenet 'conv2_block1_1_conv','conv3_block5_1_conv', 'conv2d_6','conv2d_7', 'conv2d_8', 'conv2d_9', 
    test_images_to_consider = [20, 24, 35, 39, 44]

    # model_path = '/home/pi/Projects/beyondsegmentation/Brain-tumor-segmentation/trained_models/U_resnet/ResUnet.h5'
    # weights_path = '/home/pi/Projects/beyondsegmentation/Brain-tumor-segmentation/trained_models/U_resnet/ResUnet.40_0.559.hdf5'

    # model_path = '/home/pi/Projects/BioExp/trained_models/SimUnet/Unet_without_skip.h5'
    # weights_path = '/home/pi/Projects/BioExp/trained_models/SimUnet/model_lrsch.hdf5'

    model_path = '/home/pi/Projects/BioExp/trained_models/densenet_121/densenet121.h5'
    weights_path = '/home/pi/Projects/BioExp/trained_models/densenet_121/dense_lrsch.hdf5'
   

    for model in models_to_consider:
        for layer in layers_to_consider:

            print (layer)
            D = Dissector(model_path, 
                            weights_path, 
                            "/home/pi/Projects/beyondsegmentation/HGG/**",
                            layer)

            fmap_path = 'ModelDissection_layer_fmaps_' + str(layer) + '.npy'

            if not os.path.exists(fmap_path):
                threshold_maps = D.get_threshold_maps(model, 95)

            fmaps = np.load(fmap_path)
            threshold_maps = np.percentile(fmaps, 85, axis=0)
            path = glob("/home/pi/Projects/beyondsegmentation/HGG/**")

            for image_number in range(len(path)):
                input_, label_, mask_ = load_vol(path[image_number], model, slice_= 78)
                class_filters = D.apply_threshold(input_, label_, mask_, threshold_maps, str(image_number), model, layer)
            layer_wise_filters.append(class_filters)



        layer_wise_filters = np.array(layer_wise_filters)
        ind = np.arange(len(layer_wise_filters))  # the x locations for the groups
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width, layer_wise_filters[:, 0], width,
                        label='Class 0')
        rects2 = ax.bar(ind , layer_wise_filters[:, 1], width,
                        label='Class 1')
        rects2 = ax.bar(ind + width, layer_wise_filters[:, 2], width,
                        label='Class 2')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('class specific filter count')
        # ax.set_title('Scores by group and gender')
        ax.set_xticks(ind)
        ax.set_xticklabels(layers_to_consider)
        ax.legend()
        plt.savefig(model + '_feature_count.png')

        df = pd.DataFrame()
        df['model'] = model_list
        df['filter'] = filter_list
        df['layer'] = layer_list
        df['image'] = image_list
        df['class1'] = c0_dice
        df['class2'] = c1_dice
        df['class3'] = c2_dice
        df['whole'] = c3_dice
        df['TC'] = c4_dice
        print (np.mean(c4_dice))
        # df.to_csv(model+'_dice_scores.csv')
