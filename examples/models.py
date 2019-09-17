import numpy as np 
import os
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.losses import *
from keras.callbacks import *
from keras.optimizers import *
from keras.metrics import *
from keras.utils import*
from keras.applications.densenet import DenseNet121
from keras import backend as K
# class kits_seg_model():


dropout = 0.2
bn_axis = 3
channel_axis = bn_axis

def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]
        
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))


# def categorical_focal_loss(gamma=2.0, alpha=0.25):
#     """
#     Softmax version of focal loss.
#            m
#       FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
#           c=1
#       where m = number of classes, c = class and o = observation
#     Parameters:
#       alpha -- the same as weighing factor in balanced cross entropy
#       gamma -- focusing parameter for modulating factor (1-p)
#     Default value:
#       gamma -- 2.0 as mentioned in the paper
#       alpha -- 0.25 as mentioned in the paper
#     References:
#         Official paper: https://arxiv.org/pdf/1708.02002.pdf
#         https://www org/api_docs/python/tf/keras/backend/categorical_crossentropy
#     Usage:
#      model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
#     def categorical_focal_loss_fixed(y_true, y_pred):
#         """
#         :param y_true: A tensor of the same shape as `y_pred`
#         :param y_pred: A tensor resulting from a softmax
#         :return: Output tensor.
#         """
#         gamma=2.0, 
#         alpha=0.25
#         # Scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

#         # Clip the prediction value to prevent NaN's and Inf's
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

#         # Calculate Cross Entropy
#         cross_entropy = -y_true * K.log(y_pred)

#         # Calculate Focal Loss
#         loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

#         # Sum the losses in mini_batch
#         return K.sum(loss, axis=1)

#     return categorical_focal_loss_fixed


def categorical_focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        gamma=2.0, 
        alpha=0.25
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

def softmax_dice_focal_loss(y_true, y_pred):
    return (categorical_focal_loss(y_true, y_pred) * 0.30 \
    + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.10 \
    + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.20 \
    + dice_coef_loss(y_true[..., 2], y_pred[..., 2]) * 0.20 \
    + dice_coef_loss(y_true[..., 3], y_pred[..., 3]) * 0.20)

def softmax_dice_loss(y_true, y_pred):
    return (categorical_crossentropy(y_true, y_pred) * 0.60 \
    + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.10 \
    + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.10 \
    + dice_coef_loss(y_true[..., 2], y_pred[..., 2]) * 0.10 \
    + dice_coef_loss(y_true[..., 3], y_pred[..., 3]) * 0.10)



def dice_coef_rounded_ch0(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_rounded_ch1(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_rounded_ch2(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 2]))
    y_pred_f = K.flatten(K.round(y_pred[..., 2]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_rounded_ch3(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 3]))
    y_pred_f = K.flatten(K.round(y_pred[..., 3]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def CONV2D(inputs, n_filters, filter_size,strides,padding,name):
    conv = Conv2D(n_filters, filter_size, strides=strides,
                     padding=padding,kernel_initializer='he_normal',
                     kernel_regularizer= l2(1.), name  = name)(inputs)
    conv = Dropout(rate = dropout)(conv, training=True)
    conv = Activation('relu', name=name+'_relu')(conv)
    return conv

def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = BatchNormalization(name=name, axis=bn_axis)(conv)
    conv = Dropout(rate = dropout)(conv, training=True)

    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv

def dense_conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                    padding='same',
                   use_bias=False,
                   name=name + '_1_conv')(x1)

    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Dropout(rate = dropout)(x1, training=True)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x1 = Dropout(rate = dropout)(x1, training=True)

    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = dense_conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = Dropout(rate = dropout)(x, training=True)

    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def UNET(input_shape):
    n_channels = 4
    inputs = Input(input_shape+(n_channels,))
    conv1 = CONV2D(inputs,64,3,1,'same','conv1_1')
    conv1 = CONV2D(conv1,64,3,1,'same','conv1_2')
    pool1 = MaxPooling2D(pool_size = (2,2),name = 'pool1')(conv1)

    conv2 = CONV2D(pool1,128,3,1,'same','conv2_1')
    conv2 = CONV2D(conv2,128,3,1,'same','conv2_2')
    pool2 = MaxPooling2D(pool_size = (2,2),name = 'pool2')(conv2)

    conv3 = CONV2D(pool2,256,3,1,'same','conv3_1')
    conv3 = CONV2D(conv3,256,3,1,'same','conv3_2')
    pool3 = MaxPooling2D(pool_size = (2,2),name = 'pool3')(conv3)

    conv4 = CONV2D(pool3,512,3,1,'same','conv4_1')
    conv4 = CONV2D(conv4,512,3,1,'same','conv4_2')
    pool4 = MaxPooling2D(pool_size = (2,2),name = 'pool4')(conv4)

    conv5 = CONV2D(pool4,1024,3,1,'same','conv5_1')
    conv5 = CONV2D(conv5,1024,3,1,'same','conv5_2')
    # pool5 = MaxPooling2D(pool_size = (2,2),name = 'pool5')(conv5)


    up6 = UpSampling2D(size = (2,2),name='Up6')(conv5)
    up6 = CONV2D(up6,512,3,1,'same','conv6_1')
    merged6 = concatenate([conv4, up6], axis =3)
    conv6 = CONV2D(merged6,512,3,1,'same','conv6_2')
    conv6 = CONV2D(conv6,512,3,1,'same','conv6_3')

    up7 = UpSampling2D(size = (2,2),name='Up7')(conv6)
    up7 = CONV2D(up7,256,3,1,'same','conv7_1')
    merged7 = concatenate([conv3, up7], axis =3)
    conv7 = CONV2D(merged7,256,3,1,'same','conv7_2')
    conv7 = CONV2D(conv7,256,3,1,'same','conv7_3')

    up8 = UpSampling2D(size = (2,2),name='Up8')(conv7)
    up8 = CONV2D(up8,128,3,1,'same','conv8_1')
    merged8 = concatenate([conv2, up8], axis =3)
    conv8 = CONV2D(merged8,128,3,1,'same','conv8_2')
    conv8 = CONV2D(conv8,128,3,1,'same','conv8_3')

    up9 = UpSampling2D(size = (2,2),name='Up9')(conv8)
    up9 = CONV2D(up9,64,3,1,'same','conv9_1')
    merged9 = concatenate([conv1, up9], axis =3)
    conv9 = CONV2D(merged9,64,3,1,'same','conv9_2')
    conv9 = CONV2D(conv9,64,3,1,'same','conv9_3')
    conv9 = CONV2D(conv9,3,3,1,'same','conv9_4')
    conv10 = Conv2D(4,1,1,activation = 'sigmoid',name = 'Conv10')(conv9)

    model = Model(inputs,conv10)

    return model


## Test model
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# model = UNET(input_shape=(512,512))
# model.summary()


def unet_densenet121(input_shape, weights='imagenet'):
    blocks = [6, 12, 24, 16]
    n_channel = 4
    n_class = 4
    img_input = Input(input_shape + (n_channel,))
    
    x = ZeroPadding2D(padding=((11, 11), (11, 11)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    conv1 = x
    # print(conv1)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    conv2 = x
    # print(conv2)
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    conv3 = x
    # print(conv3)
    # x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    conv4 = x
    # print(conv4)
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)
    conv5 = x 
    # print(conv5)
       
    conv6 = conv_block(UpSampling2D()(conv5), 320)
    # print(conv6)
    conv6 = concatenate([conv6,conv4], axis=-1)
    conv6 = conv_block(conv6, 320)

    conv7 = conv_block(UpSampling2D()(conv6), 256)
    # print(conv7)
    conv7 = concatenate([conv7, conv3], axis=-1)
    conv7 = conv_block(conv7, 256)

    conv8 = conv_block(UpSampling2D()(conv7), 128) 
    # print(conv8)  
    conv8 = concatenate([conv8, conv2], axis=-1)
    conv8 = conv_block(conv8, 128)

    conv9 = conv_block(UpSampling2D()(conv8), 96)
    # print(conv9)    
    conv9 = concatenate([conv9, conv1], axis=-1)
    conv9 = conv_block(conv9, 96)

    conv10 = conv_block(UpSampling2D()(conv9), 64)
    # print(conv10)
    conv10 = conv_block(conv10, 64)
    conv10 = Cropping2D(cropping = ((8,8),(8,8)))(conv10)
    res = Conv2D(n_class, (1, 1), activation='softmax', name= 'res')(conv10)
    # print(res)
    model = Model(img_input, res)
    
    # if weights == 'imagenet':
    #     densenet = DenseNet121(input_shape=input_shape + (3,), weights=weights, include_top=False)
    #     w0 = densenet.layers[2].get_weights()
    #     w = model.layers[2].get_weights()
    #     w[0][:, :, [0, 1, 2], :] = 0.9 * w0[0][:, :, :3, :]
    #     w[0][:, :, 3, :] = 0.1 * w0[0][:, :, 0, :]
    # #     # w[0][:, :, 4, :] = 0.1 * w0[0][:, :, 2, :]
    #     model.layers[2].set_weights(w)
    #     for i in range(3, len(densenet.layers)):
    #         model.layers[i].set_weights(densenet.layers[i].get_weights())
    #         model.layers[i].trainable = False
    return model

# # Test model
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# model = unet_densenet121(input_shape=(240,240))
# model.summary()


def unet_densenet121_imagenet(input_shape, weights='imagenet'):
    blocks = [6, 12, 24, 16]
    n_channel = 4
    n_class = 4
    img_input = Input(input_shape + (n_channel,))
    
    x = ZeroPadding2D(padding=((11, 11), (11, 11)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Dropout(rate = dropout)(x, training=True)
    x = Activation('relu', name='conv1/relu')(x)
    conv1 = x
    # print(conv1)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    conv2 = x
    # print(conv2)
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    conv3 = x
    # print(conv3)
    # x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    conv4 = x
    # print(conv4)
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)
    conv5 = x 
    # print(conv5)
       
    conv6 = conv_block(UpSampling2D()(conv5), 320)
    # print(conv6)
    conv6 = concatenate([conv6,conv4], axis=-1)
    conv6 = conv_block(conv6, 320)

    conv7 = conv_block(UpSampling2D()(conv6), 256)
    # print(conv7)
    conv7 = concatenate([conv7, conv3], axis=-1)
    conv7 = conv_block(conv7, 256)

    conv8 = conv_block(UpSampling2D()(conv7), 128) 
    # print(conv8)  
    conv8 = concatenate([conv8, conv2], axis=-1)
    conv8 = conv_block(conv8, 128)

    conv9 = conv_block(UpSampling2D()(conv8), 96)
    # print(conv9)    
    conv9 = concatenate([conv9, conv1], axis=-1)
    conv9 = conv_block(conv9, 96)

    conv10 = conv_block(UpSampling2D()(conv9), 64)
    # print(conv10)
    conv10 = conv_block(conv10, 64)
    conv10 = Cropping2D(cropping = ((8,8),(8,8)))(conv10)
    res = Conv2D(n_class, (1, 1), activation='softmax', name= 'res')(conv10)

    # print(res)
    model = Model(img_input, res)
    
    if weights == 'imagenet':
        densenet = DenseNet121(input_shape=input_shape + (3,), weights=weights, include_top=False)
        w0 = densenet.layers[2].get_weights()
        w = model.layers[2].get_weights()
        w[0][:, :, [0, 1, 2], :] = 0.9 * w0[0][:, :, :3, :]
        w[0][:, :, 3, :] = 0.1 * w0[0][:, :, 0, :]
    #     # w[0][:, :, 4, :] = 0.1 * w0[0][:, :, 2, :]
        model.layers[2].set_weights(w)
        counter = 0
        for i in range(3, len(densenet.layers)):
            if model.layers[i].get_config()['name'].__contains__('dropout'): 
                counter +=1
                continue

            model.layers[i].set_weights(densenet.layers[i - counter].get_weights())
            model.layers[i].trainable = False
    return model

# # Test model
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# model = unet_densenet121_imagenet(input_shape=(256,256))
# model.summary()