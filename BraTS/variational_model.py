import numpy as np
from keras.models import Model, load_model
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, GaussianNoise, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2DTranspose, UpSampling2D, concatenate, add
from keras.optimizers import SGD
import keras.backend as K
from losses_variational import *
import keras
from keras.backend.common import epsilon

K.set_image_data_format("channels_last")


# u-net model
class Unet_model_variational(object):

    def __init__(self, img_shape, load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.compile_unet()

    def compile_unet(self):
        """
        compile the U-net model
        """
        i = Input(shape=self.img_shape)
        # add gaussian noise to the first layer to combat overfitting
        i_ = GaussianNoise(0.01)(i)

        i_ = Conv2D(64, 2, padding='same', data_format='channels_last')(i_)
        i_ = Dropout(rate=0.25)(i_, training=True)
        out = self.unet(inputs=i_)
        model = Model(input=i, output=out)

        sgd = SGD(lr=0.01, momentum=0.9, decay=5e-6, nesterov=False)
        model.compile(loss=bayesian_loss, optimizer=sgd, metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        # load weights if set for prediction
        if self.load_model_weights is not None:
            model.load_weights(self.load_model_weights)
        return model

    def unet(self, inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2., activation='relu', dropout=0.25,
             batchnorm=True, upconv=True, format_='channels_last'):
        """
        the actual u-net architecture
        """
        o = self.level_block(inputs, start_ch, depth, inc_rate, activation, dropout, batchnorm, upconv, format_)
        o = BatchNormalization()(o)
        # o =  Activation('relu')(o)
        o = PReLU(shared_axes=[1, 2])(o)
        o = Conv2D(2*nb_classes, 1, padding='same', data_format=format_)(o)
        o = Dropout(rate=dropout)(o, training=True)
        o = Activation('softmax')(o)
        return o

    def level_block(self, m, dim, depth, inc, acti, do, bn, up, format_="channels_last"):
        if depth > 0:
            n = self.res_block_enc(m, 0.0, dim, acti, bn, format_)
            # using strided 2D conv for donwsampling
            m = Conv2D(int(inc * dim), 2, strides=2, padding='same', data_format=format_)(n)
            m = Dropout(rate=do)(m, training=True)
            m = self.level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, up)
            if up:
                m = UpSampling2D(size=(2, 2), data_format=format_)(m)
                m = Conv2D(dim, 2, padding='same', data_format=format_)(m)
                m = Dropout(rate=do)(m, training=True)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, padding='same', data_format=format_)(m)
            n = concatenate([n, m])
            # the decoding path
            m = self.res_block_dec(n, 0.0, dim, acti, bn, format_)
        else:
            m = self.res_block_enc(m, 0.0, dim, acti, bn, format_)
        return m

    def res_block_enc(self, m, drpout, dim, acti, bn, format_="channels_last"):

        """
        the encoding unit which a residual block
        """
        n = BatchNormalization()(m) if bn else n
        # n=  Activation(acti)(n)
        n = PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
        n = Dropout(rate=drpout)(n, training=True)
        n = BatchNormalization()(n) if bn else n
        # n=  Activation(acti)(n)
        n = PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
        n = Dropout(rate=drpout)(n, training=True)
        n = add([m, n])

        return n

    def res_block_dec(self, m, drpout, dim, acti, bn, format_="channels_last"):

        """
        the decoding unit which a residual block
        """

        n = BatchNormalization()(m) if bn else n
        # n=  Activation(acti)(n)
        n = PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
        n = Dropout(rate=drpout)(n, training=True)
        n = BatchNormalization()(n) if bn else n
        # n=  Activation(acti)(n)
        n = PReLU(shared_axes=[1, 2])(n)
        n = Conv2D(dim, 3, padding='same', data_format=format_)(n)
        n = Dropout(rate=drpout)(n, training=True)
        Save = Conv2D(dim, 1, padding='same', data_format=format_, use_bias=False)(m)
        n = add([Save, n])

        return n


def bayesian_loss(y_true, y_pred):

    shape = K.shape(y_pred)

    y_true = K.flatten(y_true)

    means = y_pred[:, :, :, :4]
    variances = y_pred[:, :, :, 4:]

    means = K.flatten(means)
    variances = K.flatten(variances)

    normalization = y_pred.shape[1]*y_pred.shape[2]

    standard_loss = K.square(y_true - means)

    updated_loss = K.sum(standard_loss * 0.5 * K.exp(-variances) + 0.5 * variances)/16384.0

    #updated_loss = K.mean(updated_loss, axis = -1)

    return updated_loss

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.
    # Returns
        Output tensor.
    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - target * tf.log(output)

    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

if __name__=='__main__':

    true = np.zeros((16, 128, 128, 4))
    pred = np.zeros((16, 128, 128, 8))

    bayesian_loss(true, pred)


