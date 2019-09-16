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
from ..helpers.utils import *

# from evaluation_metrics import *




class Dissector():
    """
        Network Dissection analysis

        model: keras model initialized with trained weights
        layer_name: intermediate layer name which needs to be analysed
    """

    def __init__(self, model, layer_name):

        self.model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        for i in range(0, len(self.model.layers)):
            self.model.layers[i].set_weights(model.layers[i].get_weights())
            self.model.layers[i].trainable = False

        self.layer_name = layer_name


    def _perform_postprocessing(self, img, threshold=800):
        """
            connected component analysis with appropreate threshold

            img: test_image for thresholding
            threshold: area threshold for selecting max area 
                    components
        """

        c,n = label(img)
        nums = np.array([np.sum(c==i) for i in range(1, n+1)])
        selected_components = np.array([threshold<num for num in nums])
        selected_components[np.argmax(nums)] = True
        mask = np.zeros_like(img)
        for i,select in enumerate(selected_components):
            if select:
                mask[c==(i+1)]=1
        return mask


    def get_threshold_maps(self, 
                            dataset_path, 
                            save_path, 
                            percentile):
        """
            Estimates threshold maps for given percentile value

            dataset_path: input dataset path
            save_path: path to save feature maps
                        if fmaps exists already it directly loads
            percentile: value used for thresholding obtained feature maps
                        range: (0, 100)
        """
        if os.path.exists(os.path.join(save_path, 'ModelDissection_layer_fmaps_{}.npy'.format(self.layer_name))):
            fmaps = np.load(os.path.join(save_path, 'ModelDissection_layer_fmaps_{}.npy'.format(self.layer_name))) 

        else:
            fmaps = []
            input_paths = glob(dataset_path)

            for i in range(len(input_paths)):
                input_, label_, _ = load_vol(input_paths[i], slice_=78)
                output = np.squeeze(self.model.predict(input_[None, ...]))
                fmaps.append(output)

            fmaps = np.array(fmaps)
            np.save(os.path.join(save_path, 'ModelDissection_layer_fmaps_{}.npy'.format(self.layer_name)), fmaps)

        threshold_maps = np.percentile(fmaps, percentile, axis=0)
            
        return threshold_maps


    def _save_features(self, img, concepts, nrows, ncols, save_path=None):
        """

        """

        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(wspace=0.025, hspace=0.05)
        
        for i in range(nrows):
            for j in range(ncols):
                try:
                    concept = concepts[:,:,i*nrows +(j+1)]
                    concept = np.ma.masked_where(concept == 0, concept)

                    ax = plt.subplot(gs[i, j], figsize=(15, 15))
                    im = ax.imshow(img[:,:,3], cmap='gray')
                    im = ax.imshow(concept, alpha=0.5)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    ax.tick_params(bottom='off', top='off', labelbottom='off' )
        
        if save_path:
            plt.savefig(os.path.join(save_path, self.layer_name+'.png'), bbox_inches='tight')
        else:
            plt.show()




    def apply_threshold(self, image, threshold_maps, 
                            nfeatures=None, 
                            save_path=None, 
                            ROI = None):
        """
            apply thresholded mask and saves the feature maps for specific iamge

            image: test image (Hx W xC)
            thresholded_maps: threshold maps used for dissection 
            nfeatures : number of features to visualize
                        all if None
            save_path : if None just displayes image else saves feature maps in 
                        given path
            ROI :  region of interest mask in a given image

        """

        fmaps = np.squeeze(self.model.predict(image[None, ...]))
        masks = fmaps >= threshold_maps
        masks = 1.*(masks)

        shape = image.shape[:-1]
        resized_masks = np.zeros((shape[0], shape[1], masks.shape[2]))
        kernel = np.ones((2, 2), np.uint8) 

        if not nfeatures:
            nfeatures = fmaps.shape[-1]


        for i in range(nfeatures):
            resized_img = imresize(masks[:,:,i], shape, interp='nearest')
            post_processed_img = self._perform_postprocessing(resized_img)
            eroded_img = (cv2.dilate(post_processed_img, kernel, iterations=1))/255
            eroded_img = eroded_img*ROI or eroded_img


        ncols = int(np.ceil(nfeatures**0.5))
        nrows = int(np.ceil(nfeatures**0.5))


        self._save_features(image, resized_masks, nrows, ncols, save_path)


    def quantify_gt_features(self, image, gt, 
                            threshold_maps, 
                            nclasses, 
                            nfeatures, 
                            save_path,
                            save_fmaps=False, 
                            ROI = None):
        """
            Quatify the learnt internal concepts by a network, 
            only valid for segmentation networks 

            image : image (H x W x C)
            gt    : image (H x W)
            threshold_maps : threshold maps used for dissection 
            nclasses : number of classes
            nfeatures : number of feature maps to consider
            save_path : path to save csv with score for each featurs
            save_fmaps: saves images with fmap overlap
            ROI : region of interest mask in a given image
        """

        fmaps = np.squeeze(self.model.predict(image[None, ...]))
        masks = fmaps >= threshold_maps
        masks = 1.*(masks)

        shape = image.shape[:-1]
        resized_masks = np.zeros((shape[0], shape[1], masks.shape[2]))
        kernel = np.ones((2, 2), np.uint8) 

        if not nfeatures:
            nfeatures = fmaps.shape[-1]

        dice_json = {}
        dice_json['feature'] = []
        for class_ in range(nclasses):
            dice_json['class_'+str(class_)] = []


        for i in range(nfeatures):
            resized_img = imresize(masks[:,:,i], shape, interp='nearest')
            post_processed_img = self._perform_postprocessing(resized_img)
            eroded_img = (cv2.dilate(post_processed_img, kernel, iterations=1))/255
            eroded_img = eroded_img*ROI or eroded_img

            dice_json['feature'].append(i)

            for class_ in range(nclasses):
                mask = gt == class_
                class_dice = (np.sum(mask*(eroded_img>0)) + 1e-5)*2.0/(np.sum(mask*1.) + np.sum((eroded_img*1.0) > 0) + 1e-5) 
                dice_json['class_'+str(class_)].append(class_dice)

            resized_masks[:,:,i] = eroded_img

        df = pd.DataFrame(dice_json)
        df.to_csv(os.path.join(save_path, self.layer_name+'_dice_scores.csv'), index=False)

        if save_fmaps:
            ncols = int(np.ceil(nfeatures**0.5))
            nrows = int(np.ceil(nfeatures**0.5))
            self._save_features(image, resized_masks, nrows, ncols, save_fmaps)
        

if __name__ == "__main__":


    path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
    path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

    test_path=path_HGG+path_LGG
    np.random.seed(2022)
    np.random.shuffle(test_path)


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
