import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb
import os
import cv2 
import keras
import random
import numpy as np
from glob import glob
import SimpleITK as sitk
import pandas as pd
from ..helpers.utils import *
from keras.models import Model
from skimage.transform import resize as imresize
from keras.utils import np_utils

import matplotlib.gridspec as gridspec
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure


class Dissector():
    """
        Network Dissection analysis

        model      : keras model initialized with trained weights
        layer_name : intermediate layer name which needs to be analysed
    """

    def __init__(self, model, layer_name, seq=None):

        self.model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        for i in range(0, len(self.model.layers)):
            self.model.layers[i].set_weights(model.layers[i].get_weights())
            self.model.layers[i].trainable = False

        self.layer_name = layer_name
        self.seq = seq


    def _perform_postprocessing(self, img, threshold=80):
        """
            connected component analysis with appropreate threshold

            img       : test_image for thresholding
            threshold : area threshold for selecting max area 
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
                            percentile,
                            loader=None):
        """
            Estimates threshold maps for given percentile value

            dataset_path: input dataset path
            save_path   : path to save feature maps
                          if fmaps exists already it directly loads
            percentile  : value used for thresholding obtained feature maps
                          range: (0, 100)
        """
        if os.path.exists(os.path.join(save_path, 'ModelDissection_layer_fmaps_{}.npy'.format(self.layer_name))):
            fmaps = np.load(os.path.join(save_path, 'ModelDissection_layer_fmaps_{}.npy'.format(self.layer_name))) 

        else:
            fmaps = []
            input_paths = os.listdir(dataset_path)

            for i in range(len(input_paths) if len(input_paths) < 500 else 500):
                print ("[INFO: BioExp] Slice no {} -- Working on {}".format(self.layer_name, i))
                input_, label_ = loader(os.path.join(dataset_path, input_paths[i]), 
					os.path.join(dataset_path, 
					input_paths[i]).replace('mask', 'label').replace('labels', 'masks'))
                output = np.squeeze(self.model.predict(input_[None, ...]))
                fmaps.append(output)

            fmaps = np.array(fmaps)

            if not os.path.exists(save_path): 
                os.makedirs(save_path)

            np.save(os.path.join(save_path, 'ModelDissection_layer_fmaps_{}.npy'.format(self.layer_name)), fmaps)

        threshold_maps = np.percentile(fmaps, percentile, axis=0)
            
        return threshold_maps


    def _save_features(self, img, concepts, nrows, ncols, save_path=None):
        """
            creats a grid of image and saves if path is given

            img : test image
            concepts: all features vectors
            nrows : number of rows in an image
            ncols : number of columns in an image
            save_path : path to save an image
        """

        plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(wspace=0.025, hspace=0.05)
        
        for i in range(nrows):
            for j in range(ncols):
                try:
                    concept = concepts[:,:,i*nrows +(j+1)]

                    concept = np.ma.masked_where(concept == 0, concept)
                    ax = plt.subplot(gs[i, j])
                    im = ax.imshow(np.squeeze(img), cmap='gray')
                    im = ax.imshow(concept, alpha=0.5)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    ax.tick_params(bottom='off', top='off', labelbottom='off' )
                except:
                    pass
        
        if save_path:
            if not os.path.exists(save_path): 
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, self.layer_name+'.png'), bbox_inches='tight')
        else:
            plt.show()




    def apply_threshold(self, image, threshold_maps, 
                            nfeatures=None, 
                            save_path=None,
                            post_process_threshold = 80, 
                            ROI = None):
        """
            apply thresholded mask and saves the feature maps for specific iamge

            image: test image (Hx W xC)
            thresholded_maps: threshold maps used for dissection 
            nfeatures : number of features to visualize
                        all if None
            save_path : if None just displayes image else saves feature maps in 
                        given path
            post_process_threshold: threshold for postprocessing cc analysis
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
            resized_img = imresize(masks[:,:,i], shape, order=0)
            try:
                post_processed_img = self._perform_postprocessing(resized_img, 
                                         threshold = post_process_threshold)
            except:
                post_processed_img = resized_img
            eroded_img = (cv2.dilate(post_processed_img, kernel, iterations=1))

            try:
                eroded_img = eroded_img*ROI 
            except: pass
            resized_masks[:,:,i] = eroded_img


        if save_path:
            ncols = int(np.ceil(nfeatures**0.5))
            nrows = int(np.ceil(nfeatures**0.5))
            self._save_features(image, resized_masks, nrows, ncols, save_path)
            
        return resized_masks


    def quantify_gt_features(self, image, gt, 
                            threshold_maps, 
                            nclasses, 
                            nfeatures, 
                            save_path,
                            save_fmaps=False, 
                            post_process_threshold=80,
                            ROI = None):
        """
            Quatify the learnt internal concepts by a network, 
            only valid for segmentation networks 

            image     : image (H x W x C)
            gt        : image (H x W)
            threshold_maps : threshold maps used for dissection 
            nclasses  : number of classes
            nfeatures : number of feature maps to consider
            save_path : path to save csv with score for each featurs
            save_fmaps: saves images with fmap overlap
            post_process_threshold: threshold for postprocessing cc analysis
            ROI       : region of interest mask in a given image
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
        for class_ in nclasses.keys():
            dice_json[class_] = []


        for i in range(nfeatures):
            resized_img = imresize(masks[:,:,i], shape, order=0)
            try:
                post_processed_img = self._perform_postprocessing(resized_img, 
                                         threshold = post_process_threshold)
            except:
                post_processed_img = resized_img
            eroded_img = (cv2.dilate(post_processed_img, kernel, iterations=1))/255
            try:
                eroded_img = eroded_img*ROI 
            except: pass

            dice_json['feature'].append(i)

            for class_ in nclasses.keys():
                mask = gt == nclasses[class_][0]
                for _class_ in nclasses[class_][1:]:
                    mask += gt == _class_
                class_dice = (np.sum(mask*(eroded_img>0)) + 1e-5)*2.0/(np.sum(mask*1.) + np.sum((eroded_img>0)*1.) + 1e-5) 
                dice_json[class_].append(class_dice)

            resized_masks[:,:,i] = eroded_img

        if not os.path.exists(save_path): 
            os.makedirs(save_path)

        df = pd.DataFrame(dice_json)
        df.to_csv(os.path.join(save_path, self.layer_name+'_dice_scores.csv'), index=False)

        if save_fmaps:
            ncols = int(np.ceil(nfeatures**0.5))
            nrows = int(np.ceil(nfeatures**0.5))
            self._save_features(image, resized_masks, nrows, ncols, save_fmaps)

        return resized_masks, df
