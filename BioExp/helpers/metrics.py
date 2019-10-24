import numpy as np
import keras.backend as K
import tensorflow as tf

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

    y_pred = tf.round(y_pred)
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    y_whole=K.sum(y_true_f[:,1:],axis=1)
    p_whole=K.sum(y_pred_f[:,1:],axis=1)
    #print(y_whole, p_whole)
    dice_whole=dice(y_whole,p_whole)
    return dice_whole

def dice_en_metric(y_true, y_pred):
    #computes the dice for the enhancing region

    y_pred = tf.round(y_pred)
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


def dice_(y_true, y_pred):
#computes the dice score on two tensors

	sum_p=K.sum(y_pred,axis=0)
	sum_r=K.sum(y_true,axis=0)
	sum_pr=K.sum(y_true * y_pred,axis=0)
	dice_numerator =2*sum_pr
	dice_denominator =sum_r+sum_p
	#print(K.get_value(2*sum_pr), K.get_value(sum_p)+K.get_value(sum_r))
	dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
	return dice_score

def metric(y_true, y_pred):
#computes the dice for the whole tumor

	y_true_f = K.reshape(y_true,shape=(-1,4))
	y_pred_f = K.reshape(y_pred,shape=(-1,4))
	y_whole=K.sum(y_true_f[:,1:],axis=1)
	p_whole=K.sum(y_pred_f[:,1:],axis=1)
	dice_whole=dice_(y_whole,p_whole)
	return dice_whole

def dice_label_metric(y_true, y_pred, label):
#computes the dice for the enhancing region
	
	y_true_f = K.reshape(y_true,shape=(-1,4))
	y_pred_f = K.reshape(y_pred,shape=(-1,4))
	y_enh=y_true_f[:,label]
	p_enh=y_pred_f[:,label]
	dice_en=dice_(y_enh,p_enh)
	return dice_en


