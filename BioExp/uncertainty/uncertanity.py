import keras
import random
import numpy as np
from glob import glob
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model

import os
import imgaug as ia
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import matplotlib.gridspec as gridspec

from ..helpers.losses import *
from .helpers.utils import load_vol_brats
# from evaluation_metrics import *


class uncertainty():
    """
    estimates model and data uncertanity

    """
    
    def __init__(self, test_image, savepath):
        """
        test_image: image for uncertanity estimation
        savepath  : path to save uncertanity images

	"""
        self.test_image = test_image
        self.savepath   = savepath

    def save(self, mean, var):
        """
        mean: mean image
        var : variance image
        """
        plt.figure(figsize=(10, 30))
        gs = gridspec.GridSpec(1, 3)
        gs.update(wspace=0.02, hspace=0.02)

        ax = plt.subplot(gs[0, 0])
        im = ax.imshow(gt.reshape((240, 240)), vmin=0., vmax=3.)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        ax = plt.subplot(gs[0, 1])
        im = ax.imshow(np.argmax(mean, axis = -1).reshape((240, 240)), vmin=0., vmax=3.)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        ax = plt.subplot(gs[0, 2])
        im = ax.imshow(var[:, :, :, 2].reshape((240, 240)), cmap=plt.cm.RdBu_r)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )

        if self.savepath:
            plt.savefig(self.savepath, bbox_inches='tight')
        else:
            plt.show()


    def epistemic(self, model, iterations = 1000):
        """
	estimates data uncertanity
        iterations: montecarlo sample iterations

        """
        self.aug = iaa.SomeOf(1, [
                        iaa.Affine(
                        rotate=iap.Normal(0.0, 3),
                        translate_px=iap.Normal(0.0, 3)),
                        # iaa.AdditiveGaussianNoise(scale=0.3 * np.ptp(test_image) - 9),
                        iaa.Noop(),
                        iaa.MotionBlur(k=3, angle = [-2, 2])
                    ], random_order=True)

        predictions = []
        
        for i in range(iterations):
            aug_image = self.aug.augment_images(self.test_image)
            predictions.append(model.predict(aug_image[None, ...]))
            
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis = 0)
        var = np.var(predictions, axis = 0)
        
        return mean, var


    def aleatoric(self, model, iterations=1000, dropout=0.5):
        """
	estimates model uncertanity
        iterations: montecarlo sample iterations

        """
        predictions = []
        
        for i in range(iterations):
            predictions.append(model.predict(self.test_image[None, ...]))
            
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis = 0)
        var = np.var(predictions, axis = 0)
        if np.sum(var) == 0: raise ValueError("Model trained without dropouts")
        

        return mean, var

    def combined(self, model, iterations=1000, dropout=0.5):
        """
	estimates combined uncertanity
        iterations: montecarlo sample iterations

        """
        self.aleatoric(model, iterations=1)
        return self.epistemic(model, iterations)



