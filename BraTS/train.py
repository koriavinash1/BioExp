import numpy as np
import random
import json
from glob import glob
from keras.models import model_from_json,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint,Callback,LearningRateScheduler
import keras.backend as K
from model import Unet_model
from variational_model import Unet_model_variational
from model_simple import Unet_model_simple
from losses_variational import *
#from keras.utils.visualize_util import plot
from extract_patches import *
from model import Unet_model
from data_generator import DataGenerator

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))



class SGDLearningRateTracker(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        if epoch%10 == 0 and epoch !=0:
            optimizer = self.model.optimizer
            lr = K.get_value(optimizer.lr)
            decay = K.get_value(optimizer.decay)
            lr=lr/10
            decay=decay*10
            K.set_value(optimizer.lr, lr)
            K.set_value(optimizer.decay, decay)
            print('LR changed to:',lr)
            print('Decay changed to:',decay)



class Training(object):
    
    def __init__(self, batch_size, nb_epoch,load_model_resume_training=None):

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        #loading model from path to resume previous training without recompiling the whole model
        if load_model_resume_training is not None:
            self.model =load_model(load_model_resume_training,custom_objects={'gen_dice_loss': gen_dice_loss,
                                                                        'dice_whole_metric':dice_whole_metric,
                                                                        'dice_core_metric':dice_core_metric,
                                                                        'dice_en_metric':dice_en_metric})
            print("pre-trained model loaded!")
        else:
            unet = Unet_model(img_shape=(240, 240, 1))
            self.model= unet.model
            #self.model.load_weights('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/Unet_cc/SimUnet.01_0.095.hdf5')
            print("U-net CNN compiled!")

    def fit(self, train_gen, val_gen):

        train_generator = train_gen
        val_generator = val_gen
        checkpointer = ModelCheckpoint(filepath='../results/BraTs/checkpoints/Unet/model.{epoch:02d}_{val_loss:.3f}.hdf5', verbose=1, period = 5)
        self.model.fit_generator(train_generator,
                                 epochs=self.nb_epoch, steps_per_epoch=100, validation_data=val_generator, validation_steps=100,  verbose=1,
                                 callbacks=[checkpointer, SGDLearningRateTracker()])


    def img_msk_gen(self,X33_train,Y_train,seed):

        '''
        a custom generator that performs data augmentation on both patches and their corresponding targets (masks)
        '''
        datagen = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
        datagen_msk = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
        image_generator = datagen.flow(X33_train,batch_size=self.batch_size,seed=seed)
        y_generator = datagen_msk.flow(Y_train,batch_size=self.batch_size,seed=seed)
        while True:
            yield(image_generator.next(), y_generator.next())


    def save_model(self, model_name):
        '''
        INPUT string 'model_name': path where to save model and weights, without extension
        Saves current model as json and weights as h5df file
        '''

        model_tosave = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model_tosave, 'w') as f:
            json.dump(json_string, f)
        print ('Model saved.')

    def load_model(self, model_name):
        '''
        Load a model
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        print ('Loading model {}'.format(model_name))
        model_toload = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model_toload) as f:
            m = next(f)
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)
        print ('Model loaded.')
        self.model = model_comp
        return model_comp

import os
import psutil
import timeit
import gc

def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()

if __name__ == "__main__":
    #set arguments

    #reload already trained model to resume training
    model_to_load="Models/ResUnet.04_0.646.hdf5" 
    #save=None

    #compile the model
    brain_seg = Training(batch_size=16, nb_epoch=100)

    print(brain_seg.model.summary())

    train_generator = DataGenerator('../sample_vol/', batch_size=16)
    val_generator = DataGenerator('../sample_vol/', batch_size=16)
    
    brain_seg.fit(train_generator, val_generator)
    brain_seg.model.save('../results/BraTs/checkpoints/Unet/model_archi.h5')
    #random.seed(7)
