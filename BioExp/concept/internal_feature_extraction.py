import lucid.optvis.param as param
import lucid.optvis.render as render
from lucid.misc.io.showing import _image_url, _display_html
from lucid.modelzoo.vision_base import Model
import tensorflow as tf
import lucid.optvis.transform as transform
from lucid.optvis import objectives
from lucid.misc.io import show
from lucid.optvis.objectives_util import _dot, _dot_cossim, _extract_act_pos, _make_arg_str, _T_force_NHWC, _T_handle_batch
from lucid.optvis.objectives import Objective
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pprint import pprint
import matplotlib.gridspec as gridspec
from decorator import decorator

class Feature_Visualizer():

  def __init__(self, model_loader, layer=None, channel=None, regularizer_params=dict.fromkeys(['jitter', 'rotate', 'scale', 'TV', 'blur', 'decorrelate', 'L1'])):

    default_dict = dict.fromkeys(['jitter', 'rotate', 'scale', 'TV', 'blur', 'decorrelate', 'L1'])
    for key in regularizer_params.keys():
      default_dict[key] = regularizer_params[key] 
    regularizer_dict = default_dict

    self.loader = model_loader
    self.jitter = regularizer_params['jitter'] or 8
    self.rotate = regularizer_params['rotate'] or 10
    self.scale = regularizer_params['scale'] or 1.2
    self.TV = regularizer_params['TV'] or -5e-7
    self.blur = regularizer_params['blur'] or 0
    self.decorrelate = regularizer_params['decorrelate'] or True
    self.L1 = regularizer_params['L1'] or 1e-4
    self.layer = layer or -1
    self.channel = channel or 0

  def show_images(self, images):
    html = ""
    for image in images:
      data_url = _image_url(image)
      html += '<img width=\"100\" style=\"margin: 10px\" src=\"' + data_url + '\">'
    _display_html(html)


  def wrap_objective(require_format=None, handle_batch=False):
    """Decorator for creating Objective factories.

    Changes f from the closure: (args) => () => TF Tensor
    into an Obejective factory: (args) => Objective

    while perserving function name, arg info, docs... for interactive python.
    """ 
    @decorator
    def inner(f, *args, **kwds):
      objective_func = f(*args, **kwds)
      objective_name = f.__name__
      args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
      description = objective_name.title() + args_str

      def process_T(T):
        if require_format == "NHWC":
          T = _T_force_NHWC(T)
        return T

      return Objective(lambda T: objective_func(process_T(T)),
                       objective_name, description)
    return inner


  @wrap_objective(require_format='NHWC')
  def _channel(self, layer, n_channel, gram = None):
    """Visualize a single channel"""

    def inner(T):
      if gram is not None:
        kernel = lambda x, y: tf.reduce_mean(tf.exp((-1./(2*2**2))*tf.abs(x-y)**2))
        
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
        var_vec = tf.reshape(var, [-1, 4])
        gram_vec = tf.reshape(gram, [-1, 4])

        kernel_loss = 0
        for i in range(4):
          for j in range(4):
            #print ("calculating kernal loss")
            kernel_loss  += kernel(var_vec[:, i], var_vec[:, j]) + kernel(gram_vec[:, i], gram_vec[:, j]) - 2*kernel(var_vec[:, i], gram_vec[:, j])
        
        return tf.reduce_mean(T(layer)[..., n_channel]) + 1e-2*kernel_loss + self.L1*tf.norm(var) 
      else:
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
        return tf.reduce_mean(T(layer)[..., n_channel]) + self.L1*tf.norm(var) 
    return inner

  
  def run(self, savepath, n_channels=4, style_template=None):

    model = self.loader()
    model.load_graphdef()

    # layer_to_consider = ['conv2d_3', 'conv2d_5', 'conv2d_7', 'conv2d_13', 'conv2d_15', 'conv2d_17',  'conv2d_21', 'conv2d_23', 'conv2d_25']

    with tf.Graph().as_default() as graph, tf.Session() as sess:

      if style_template is not None:
        gram_template = tf.constant(np.load(style_template), #[1:-1,:,:],
                                    dtype=tf.float32) 

      obj = self._channel(self.layer+"/convolution", self.channel, gram=style_template)
      obj += self.TV * objectives.total_variation()
      obj += self.blur * objectives.blur_input_each_step()

      transforms = [
        transform.pad(2 * self.jitter),
        transform.jitter(self.jitter),
        # transform.random_scale([self.scale ** (n/10.) for n in range(-10, 11)]),
        transform.random_rotate(range(-self.rotate, self.rotate + 1))
      ]

      T = render.make_vis_T(model, obj,
                            param_f=lambda: param.image(240, channels=n_channels, fft=self.decorrelate,
                                                        decorrelate=self.decorrelate),
                            optimizer=None,
                            transforms=transforms, relu_gradient_override=True)
      tf.initialize_all_variables().run()

      # pprint([v.name for v in tf.get_default_graph().as_graph_def().node])
      for i in range(400):
        T("vis_op").run()

      plt.figure(figsize=(30,10))

      for i in range(1, 5):
        plt.subplot(1, 4, i)
        image = T("input").eval()[:, :, :, i - 1].reshape((240, 240))
        print(image.min(), image.max())
        plt.imshow(T("input").eval()[:, :, :, i - 1].reshape((240, 240)), cmap='gray',
                   interpolation='bilinear', vmin=0., vmax=1.)
        plt.xticks([])
        plt.yticks([])

        # show(np.hstack(T("input").eval()))
    plt.savefig(savepath+self.layer+'_' + str(self.channel) +'.png', bbox_inches='tight')


if __name__ == '__main__':

  # Initialize a class which loads a Lucid Model Instance with the required parameters
  class Load_Model(Model):

    model_path = '/home/parth/lucid/lucid/model_res.pb'
    image_shape = [None, 4, 240, 240]
    image_value_range = (0, 1)
    input_name = 'input_1'

  # Initialise a Visualizer instance
  E = Feature_Visualizer(Load_Model, 'conv2d_23', 0)

  # Run the Visualizer
  E.run('stylereg_features_', n_channels=4)