import tensorflow as tf
import keras.backend as K

resnet_model_path = 'trained_models/U_resnet/ResUnet.h5'
resnet_weights_path = 'trained_models/U_resnet/ResUnet.15_0.491.hdf5'
resnet_pb_path = 'trained_models/U_resnet/resnet.pb'


sunet_model_path = 'trained_models/SimUnet/FCN.h5'
sunet_weights_path = 'trained_models/SimUnet/SimUnet.40_0.060.hdf5'
sunet_pb_path = 'trained_models/SimUnet/SUnet.pb'


dense_model_path = 'trained_models/densenet_121/densenet121.h5'
dense_weights_path = 'trained_models/densenet_121/densenet.55_0.522.hdf5'
dense_pb_path = 'trained_models/densenet_121/densenet.pb'

shallow_model_path = 'trained_models/shallowunet/shallow_unet.h5'
shallow_weights_path = 'trained_models/shallowunet/shallow_weights.hdf5'
shallow_pb_path = 'trained_models/shallowunet/shallow_unet.pb'


from keras.models import load_model
from models import *
from losses import *


def load_seg_model(model_='shallow'):
    
#     model = unet_densenet121_imagenet((240, 240), weights='imagenet12')
#     model.load_weights(weights_path)
    
    if model_ == 'uresnet':
        model = load_model(resnet_model_path, custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(resnet_weights_path)
        return model, resnet_weights_path, resnet_pb_path
    
    elif model_ == 'fcn':
        model = load_model(sunet_model_path, custom_objects={'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(sunet_weights_path)
        return model, sunet_weights_path, sunet_pb_path
    
    elif model_  == 'dense':
        model = load_model(dense_model_path, custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(dense_weights_path)
        return model, dense_weights_path, dense_pb_path
   
    elif model_ == 'shallow':
        model = load_model(shallow_model_path, custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(shallow_weights_path)
        return model, shallow_weights_path, shallow_pb_path

def save_frozen_graph(filename):
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
          session,
          K.get_session().graph.as_graph_def(),
          ['conv2d_32/BiasAdd']
      )
    with open(filename, "wb") as f:
          f.write(output_graph_def.SerializeToString())


with tf.Session(graph=K.get_session().graph) as session:
    session.run(tf.global_variables_initializer())
    
    model_res, weights_path, pb_path = load_seg_model()
    print (model_res.summary())
    save_frozen_graph(pb_path)

import tensorflow as tf
graph_def = tf.GraphDef()

with open(pb_path, "rb") as f:
    graph_def.ParseFromString(f.read())
    
for node in graph_def.node:
    print(node.name)
