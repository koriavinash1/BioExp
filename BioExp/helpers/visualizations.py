import keras
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2
import os
from keras import backend as K
from keras import activations
from keras.models import Model

import tensorflow as tf
from tensorflow.python.framework import ops


normalize = lambda x: (x + K.epsilon()) / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def denormalize(x):
    r"""
        Denormalize the image values

        converts the range of image intensity 
        values from (0,1) to (0, 255)

    """
    x = ((x-x.mean())/(x.std()+K.epsilon())*0.25) + 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _guided_backprop_(model, seed_input, start_layer, end_layer):
    r"""
        Guided Backpropagation method for visualizing input saliency.
        gradient will be taken as followes:
            $$\frac{\partial model[end_layer]}{\partial model[start_layer]}$$

        model: <keras model>
        seed_input: <ndarray>; input image for a network
        start_layer: <int>; layer index
        end_layer: <int>; layer index

    """

    layer_input = model.layers[start_layer].output
    layer_output = model.layers[end_layer].output

    grads = K.gradients(layer_output, layer_input)[0]
    backprop_fn = K.function([layer_input, K.learning_phase()], [grads])
    grads_val = backprop_fn([seed_input, 0])[0]
    return grads_val


def _gradCAM_(model, seed_input, start_layer, end_layer, cls=0, normalize = False):
    r"""
        applies CAM over each input

        model: <keras model>
        seed_input: <ndarray>; input image for a network
        start_layer: <int>; layer index
        end_layer: <int>; layer index
        cls: <int>; output class, 0 by default
        normalize: <bool> for gradient normalization
    """

    y_c = model.layers[end_layer].output[0, cls]
    y_c = K.mean(y_c)
    conv_output = model.layers[start_layer].output
    grads = K.gradients(y_c, conv_output)[0]

    if normalize: grads = normalize(grads)
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    W = seed_input[0].shape[0]; H = seed_input[0].shape[1]
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    return cam


def visualize_cam(model, 
                st_layer_idx, 
                filter_indices, 
                penultimate_layer_idx,  
                seed_input, 
                backprop_modifier = None):

    r"""
        generated class activation map as proposed in:
        https://arxiv.org/abs/1610.02391

         gradient will be taken as followes:
            $$\frac{\partial model[penultimate_layer_idx]}{\partial model[st_layer_idx]}$$

        variable names are taken from keras-vis
        
        model: <keras model>; keras trained model
        st_layer_idx: <int>; layer index
        filter_indices: <list>; list of class idx used in analysis
        penultimate_layer_idx: <int>; layer index
        seep_input: <ndarray>; input image with batch axis
        backprop_modifier: <None or str>; None by default, allowed ['guided'] 

    """

    CAM = _gradCAM_(model, seed_input, st_layer_idx, penultimate_layer_idx, filter_indices)

    if backprop_modifier:
        if not backprop_modifier == 'guided':
            raise ValueError("[INFO: BioExp Helpers] allowed backprop_modifier are [None, 'guided']")
        else:
            if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * \
                       tf.cast(op.inputs[0] > 0., dtype)

            g = tf.get_default_graph()
            with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
                model = Model(model.inputs, model.outputs)
            guided  = _guided_backprop_(model,seed_input, st_layer_idx, penultimate_layer_idx)
            guidedCAM = guided*CAM[..., None]
            return denormalize(guided)


    CAM = cv2.applyColorMap(np.uint8(255 * CAM), cv2.COLORMAP_JET)
    return np.uint8(CAM)


