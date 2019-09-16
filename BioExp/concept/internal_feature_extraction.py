import lucid.optvis.param as param
import lucid.optvis.render as render
from lucid.misc.io.showing import _image_url, _display_html
from lucid.modelzoo.vision_base import Model
import tensorflow as tf
import lucid.optvis.transform as transform
from lucid.optvis import objectives
from lucid.misc.io import show
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pprint import pprint
import matplotlib.gridspec as gridspec

def show_images(images):
  html = ""
  for image in images:
    data_url = _image_url(image)
    html += '<img width=\"100\" style=\"margin: 10px\" src=\"' + data_url + '\">'
  _display_html(html)

class Tumor_Mode(Model):
  model_path = '/home/pi/Projects/beyondsegmentation/Brain-tumor-segmentation/trained_models/U_resnet/resnet.pb'
  image_shape = [None, 4, 240, 240]
  image_value_range = (0, 1)
  input_name = 'input_1'


def channel(layer, n_channel, batch=None, gram = None):
  """Visualize a single channel"""

  def inner(T):
    kernel = lambda x, y: tf.reduce_mean(tf.exp((-1./(2*2**2))*tf.abs(x-y)**2))
    
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
    var_vec = tf.reshape(var, [-1, 4])
    gram_vec = tf.reshape(gram, [-1, 4])

    kernel_loss = 0
    for i in range(4):
      for j in range(4):
        print ("calculating kernal loss")
        kernel_loss  += kernel(var_vec[:, i], var_vec[:, j]) + kernel(gram_vec[:, i], gram_vec[:, j]) - 2*kernel(var_vec[:, i], gram_vec[:, j])
    
    return tf.reduce_mean(T(layer)[..., n_channel]) + 1e-2*kernel_loss # + 1e-4*tf.norm(var) # + 1e-0*tf.sqrt(epsilon + tf.reduce_mean((gram - image_gram) ** 2))
  return inner


tumor_model = Tumor_Mode()
tumor_model.load_graphdef()

JITTER = 8
ROTATE = 10
SCALE = 1.
L1 = -0.05
TV = -0.25
BLUR = -1.0

DECORRELATE = True

# layer_to_consider = ['conv2d_3', 'conv2d_5', 'conv2d_7', 'conv2d_13', 'conv2d_15', 'conv2d_17',  'conv2d_21', 'conv2d_23', 'conv2d_25']
layer_to_consider = ['conv2d_17']
channel_idx = 31
for layer in layer_to_consider:
  with tf.Graph().as_default() as graph, tf.Session() as sess:

    gram_template = tf.constant(np.load('/home/pi/Projects/beyondsegmentation/Brain-tumor-segmentation/test_image.npy'), #[1:-1,:,:],
                                dtype=tf.float32)


    obj = channel(layer+"/convolution", channel_idx, gram=gram_template)
    # obj += L1 * objectives.L1(constant=.5)
    # obj += TV * objectives.total_variation()
    # obj += BLUR * objectives.blur_input_each_step()

    transforms = [
      transform.pad(2 * JITTER),
      transform.jitter(JITTER),
      # transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
      transform.random_rotate(range(-ROTATE, ROTATE + 1))
    ]

    T = render.make_vis_T(tumor_model, obj,
                          param_f=lambda: param.image(240, channels=4, fft=DECORRELATE,
                                                      decorrelate=DECORRELATE),
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
  plt.savefig('stylereg_features_'+layer+'_' + str(channel_idx) +'.png', bbox_inches='tight')

