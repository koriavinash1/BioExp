from lucid.modelzoo.vision_base import Model
import sys
sys.path.append('..')
from BioExp.concept.feature import Feature_Visualizer


# Initialize a class which loads a Lucid Model Instance with the required parameters
class Load_Model(Model):

  model_path = '../../saved_models/model_flair/flair.pb'
  image_shape = [None, 1, 240, 240]
  image_value_range = (0, 1)
  input_name = 'input_1'

# Initialize a Visualizer Instance
E = Feature_Visualizer(Load_Model, savepath = '../results/')

# Run the Visualizer
E.run(layer = 'conv2d_3', channel = 4)
