from lucid.modelzoo.vision_base import Model
import sys
sys.path.append('..')
from BioExp.concept.feature import Feature_Visualizer


# Initialize a class which loads a Lucid Model Instance with the required parameters
class Load_Model(Model):

  model_path = '/home/parth/lucid/lucid/model_res.pb'
  image_shape = [None, 4, 240, 240]
  image_value_range = (0, 1)
  input_name = 'input_1'

# Initialize a Visualizer Instance
E = Feature_Visualizer(Load_Model, 'conv2d_23', channel = 0, savepath = '../results/')

# Run the Visualizer
E.run()