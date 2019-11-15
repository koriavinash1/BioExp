import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from .losses import *

def generate_pb(model_path, layer_name, pb_path, wts_path):
    """
    freezes model weights and convert entire graph into .pb file

    model_path: saved model path (model architecture) (str) 
    layer_name: name of output layer (str)
    pb_path   : path to save pb file
    wts_path  : saved model weights
    """
    with tf.Session(graph=K.get_session().graph) as session:

        session.run(tf.global_variables_initializer())
        model = load_model(model_path, custom_objects={'gen_dice_loss':gen_dice_loss,
                                        'dice_whole_metric':dice_whole_metric,
                                        'dice_core_metric':dice_core_metric,
                                        'dice_en_metric':dice_en_metric})
        model.load_weights(wts_path)
        print (model.summary())
        try:
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                session,
                K.get_session().graph.as_graph_def(),
                [layer_name + '/convolution']
            )
        except:
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                session,
                K.get_session().graph.as_graph_def(),
                [layer_name + '/convolution']
            )
        with open(pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
