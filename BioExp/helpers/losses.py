import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils

def dice(y_true, y_pred):
    #computes the dice score on two tensors
    #y_pred = tf.round(y_pred)

    sum_p=K.sum(y_pred,axis=0)
    sum_r=K.sum(y_true,axis=0)
    sum_pr=K.sum(y_true * y_pred,axis=0)
    dice_numerator =2*sum_pr
    dice_denominator =sum_r+sum_p
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score

def dice_updated(y_true, y_pred):
    #computes the dice score on two tensors
    #y_pred = tf.round(y_pred)

    sum_p=K.sum(y_pred,axis=[1,2])
    sum_r=K.sum(y_true,axis=[1,2])
    sum_pr=K.sum(y_true * y_pred,axis=[1,2])
    dice_numerator =2*sum_pr
    dice_denominator =sum_r+sum_p
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score


def dice_whole_metric(y_true, y_pred):
    #computes the dice for the whole tumor

    #y_pred = tf.round(y_pred)
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    y_whole=K.sum(y_true_f[:,1:],axis=1)
    p_whole=K.sum(y_pred_f[:,1:],axis=1)
    #print(y_whole, p_whole)
    dice_whole=dice(y_whole,p_whole)
    return dice_whole

def dice_en_metric(y_true, y_pred):
    #computes the dice for the enhancing region

    #y_pred = tf.round(y_pred)
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    y_enh=y_true_f[:,-1]
    p_enh=y_pred_f[:,-1]
    dice_en=dice(y_enh,p_enh)
    return dice_en

def dice_core_metric(y_true, y_pred):
    ##computes the dice for the core region

    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    
    #workaround for tf
    y_core=K.sum(tf.gather(y_true_f, [1,3],axis =1),axis=1)
    p_core=K.sum(tf.gather(y_pred_f, [1,3],axis =1),axis=1)
    
    #y_core=K.sum(y_true_f[:,[1,3]],axis=1)
    #p_core=K.sum(y_pred_f[:,[1,3]],axis=1)
    dice_core=dice(y_core,p_core)
    return dice_core



def weighted_log_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # weights are assigned in this order : normal,necrotic,edema,enhancing 
    weights=np.array([1,5,2,4])
    weights = K.variable(weights, name='weights')
    loss = y_true * K.log(y_pred) * weights
    loss = K.mean(-K.sum(loss, -1))
    return loss

def gen_dice_loss(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''

    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights

    return GDL+weighted_log_loss(y_true,y_pred)

def dice_loss(y_true, y_pred):
    return(1-dice(y_true, y_pred))


def soft_dice_loss(y_true, y_pred):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    #y_pred = keras.utils.np_utils.to_categorical(np.argmax(y_true, axis = -1), num_classes=y_pred.shape[-1])
    epsilon = 1e-6

    # skip the batch not class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)))
    #print(axes)
    numerator = 2. * np.sum(y_pred * y_true, axis=0)
    denominator = np.sum(y_pred + y_true, axis=0)

    #print(numerator, denominator)
    return np.mean(numerator / (denominator + epsilon))  # average over classes and batch

smooth = 1e-3

def dice_core_coef(y_true, y_pred):

    y_true = np.reshape(y_true, (-1, 4))
    y_pred = np.reshape(y_pred, (-1, 4))

    y_whole = np.sum(y_true[:, [1,3]], axis = 1)
    p_whole = np.sum(y_pred[:, [1,3]], axis= 1)
    #print(y_whole.shape)
    return(dice_coef(y_whole, p_whole))


def dice_en_coef(y_true, y_pred):

    y_true = np.reshape(y_true, (-1, 4))
    y_pred = np.reshape(y_pred, (-1, 4))

    y_whole = y_true[:, -1]
    p_whole = y_pred[:, -1]
    #print(y_whole.shape)
    return(dice_coef(y_whole, p_whole))

def dice_whole_coef(y_true, y_pred):
 
    y_true = np_utils.to_categorical(y_true, num_classes=4)
    y_pred = np_utils.to_categorical(y_pred, num_classes=4)

    y_true = np.reshape(y_true, (-1, 4))
    y_pred = np.reshape(y_pred, (-1, 4))

    y_whole = np.sum(y_true[:, 1:], axis = 1)
    p_whole = np.sum(y_pred[:, 1:], axis= 1)
    #print(y_whole.shape)
    return(dice_coef(y_whole, p_whole))


def dice_label_coef(y_true, y_pred, labels, n_classes=4):

    y_true = np_utils.to_categorical(y_true, num_classes=n_classes)
    y_pred = np_utils.to_categorical(y_pred, num_classes=n_classes)

    y_true = np.reshape(y_true, (-1, n_classes))
    y_pred = np.reshape(y_pred, (-1, n_classes))
    
    y_whole = np.sum(y_true[:, np.array(labels, dtype='int64')], axis = 1)
    p_whole = np.sum(y_pred[:, np.array(labels, dtype='int64')], axis = 1)

    return(dice_coef(y_whole, p_whole))


def dice_coef(y_true, y_pred):

    intersection = np.sum(y_true * y_pred, axis = 0)

    sum_r= np.sum(y_true, axis = 0)
    sum_p = np.sum(y_pred, axis =  0)
    #print(intersection)
    return (2. * intersection + smooth) / (sum_p + sum_r + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
