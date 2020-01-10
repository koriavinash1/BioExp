import matplotlib
# matplotlib.use('Agg')
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

from lucid.optvis.param.color import to_valid_rgb
from lucid.optvis.param.spatial import pixel_image, fft_image
import os
import sys
sys.path.append('../..')
# from BioExp.helpers import transform

class Feature_Visualizer():
  """
  A class for generating Feature Visualizations of internal filters from a .pb model file (based on Lucid)

  Inputs: model_loader: A function that loads an instance of the Lucid Model class (see examples)
          layer: The layer to visualize
          channel(optional): The index of the filter to visualize; defaults to 0.
          savepath(optional): Path to save visualized features
          n_channels(optional): Number of channels in model input; defaults to 4
          regularizer_params(optional): A dictionary of regularizer parameters for the optimizer. Parameters which are not given will default to below values.

  Outputs: Visualized Feature saved at savepath
  """

  def __init__(self, model_loader, savepath = './', n_channels = 4, regularizer_params=dict.fromkeys(['jitter', 'rotate', 'scale', 'TV', 'blur', 'decorrelate', 'L1'])):

    default_dict = dict.fromkeys(['jitter', 'rotate', 'scale', 'TV', 'blur', 'decorrelate', 'L1'])
    for key in regularizer_params.keys():
      default_dict[key] = regularizer_params[key] 
    regularizer_dict = default_dict
    print('Regularizer Paramaters: ', regularizer_dict)

    self.loader = model_loader
    self.jitter = regularizer_dict['jitter'] if regularizer_dict['jitter'] is not None else 8
    self.rotate = regularizer_dict['rotate'] if regularizer_dict['rotate'] is not None else 4
    self.scale = regularizer_dict['scale'] if regularizer_dict['scale'] is not None else 1.2
    self.TV = regularizer_dict['TV'] if regularizer_dict['TV'] is not None else 0
    self.blur = regularizer_dict['blur'] if regularizer_dict['blur'] is not None else 0
    self.decorrelate = regularizer_dict['decorrelate'] if regularizer_dict['decorrelate'] is not None else True
    self.L1 = regularizer_dict['L1'] if regularizer_dict['L1'] is not None else 1e-5
    self.savepath = savepath
    self.n_channels = n_channels
    
    print("jitter: {}, rotate: {}".format(self.jitter, self.rotate))
    self.model = self.loader()
    self.model.load_graphdef()  

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
  def _channel(self, layer, n_channel, gram = None, gram_coeff = 1e-4):
    """Visualize a single channel"""

    def inner(T):
      if gram is not None:
        kernel = lambda x, y: tf.reduce_mean(tf.exp((-1./(2*2**2))*tf.abs(x-y)**2))
        
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
        var_vec = tf.reshape(var, [-1, 4])
        print(gram.get_shape().as_list())
        gram_vec = tf.reshape(gram, [-1, 4])
        

        kernel_loss = 0
        for i in range(4):
          for j in range(4):
            kernel_loss  += kernel(var_vec[:, i], var_vec[:, j]) + kernel(gram_vec[:, i], gram_vec[:, j]) - 2*kernel(var_vec[:, i], gram_vec[:, j])

        # kernel_loss = tf.math.abs((var_vec - gram_vec)**2)
        return tf.reduce_mean(T(layer)[..., n_channel]) - gram_coeff*kernel_loss - self.L1*tf.norm(var) 

      else:
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
        return tf.reduce_mean(T(layer)[..., n_channel]) # + tf.math.reduce_std(var)  - self.L1*tf.norm(var) 
    return inner

  
  def image(
      self,
      w,
      h=None,
      batch=None,
      sd=None,
      decorrelate=True,
      fft=True,
      alpha=False,
      channels=None,
  ):
      h = h or w
      batch = batch or 1
      ch = channels or (4 if alpha else 3)
      shape = [batch, h, w, ch]
      param_f = fft_image if fft else pixel_image
      t = param_f(shape, sd=sd)
      if channels:
          output = tf.nn.sigmoid(t)
      else:
          output = to_valid_rgb(t[..., :3], decorrelate=decorrelate, sigmoid=True)
          if alpha:
              a = tf.nn.sigmoid(t[..., 3:])
              output = tf.concat([output, a], -1)
      return output

  def run(self, layer, class_, channel=None, style_template=None, transforms = False, opt_steps = 500, gram_coeff = 1e-14):
    """
    layer         : layer_name to visualize
    class_        : class to consider
    style_template: template for comparision of generated activation maximization map
    transforms    : transforms required
    opt_steps     : number of optimization steps
    """

    self.layer = layer
    self.channel = channel if channel is not None else 0
    
    with tf.Graph().as_default() as graph, tf.Session() as sess:

      if style_template is not None:

        try:
          gram_template = tf.constant(np.load(style_template), #[1:-1,:,:],
                                      dtype=tf.float32) 
        except:
          image = cv2.imread(style_template)
          print(image.shape)
          gram_template = tf.constant(np.pad(cv2.imread(style_template), ((1, 1), (0, 0))), #[1:-1,:,:],
                                      dtype=tf.float32) 
      else:
        gram_template = None

      obj  = self._channel(self.layer+"/convolution", self.channel, gram=gram_template, gram_coeff = gram_coeff)
      obj += -self.L1 * objectives.L1(constant=.5)
      obj += -self.TV * objectives.total_variation()
      #obj += self.blur * objectives.blur_input_each_step()


      if transforms == True:
        transforms = [
          transform.pad(self.jitter),
          transform.jitter(self.jitter),
          #transform.random_scale([self.scale ** (n/10.) for n in range(-10, 11)]),
          #transform.random_rotate(range(-self.rotate, self.rotate + 1))
        ]
      else:
        transforms = []
      
      T = render.make_vis_T(self.model, obj,
                            param_f=lambda: self.image(240, channels=self.n_channels, fft=self.decorrelate,
                                                        decorrelate=self.decorrelate),
                            optimizer=None,
                            transforms=transforms, relu_gradient_override=False)
      tf.initialize_all_variables().run()

      for i in range(opt_steps):
        T("vis_op").run()

      plt.figure(figsize=(10,10))
      # for i in range(1, self.n_channels+1):
      #   plt.imshow(np.load(style_template)[:, :, i-1], cmap='gray',
      #              interpolation='bilinear', vmin=0., vmax=1.)
      #   plt.savefig('gram_template_{}.png'.format(i), bbox_inches='tight')
        
      texture_images = []

      for i in range(1 self.n_channels+1):
        # plt.subplot(1, self.n_channels, i)
        image = T("input").eval()[:, :, :, i - 1].reshape((240, 240))
        print("channel: ", i, image.min(), image.max())
        # plt.imshow(image, cmap='gray',
        #            interpolation='bilinear', vmin=0., vmax=1.)
        # plt.xticks([])
        # plt.yticks([])
        texture_images.append(image)
        # show(np.hstack(T("input").eval()))

        os.makedirs(os.path.join(self.savepath, class_), exist_ok=True)
        # print(self.savepath, class_, self.layer+'_' + str(self.channel) +'.png')
        # plt.savefig(os.path.join(self.savepath, class_, self.layer+'_' + str(self.channel) + '_' + str(i) +'_noreg.png'), bbox_inches='tight')
      # plt.show()
      # print(np.array(texture_images).shape) 

    return np.array(texture_images)

