from lucid.modelzoo.vision_base import Model
import sys
sys.path.append('..')
from BioExp.concept.feature import Feature_Visualizer


# Initialize a class which loads a Lucid Model Instance with the required parameters
class Load_Model(Model):

  model_path = '../../saved_models/model_flair/model.pb'
  image_shape = [None, 1, 240, 240]
  image_value_range = (0, 10)
  input_name = 'input_1'

# Initialize a Visualizer Instance
E = Feature_Visualizer(Load_Model, savepath = '../results/', regularizer_params={'L1':1e-4, 'rotate':10})

# Run the Visualizer
a = E.run(layer = 'conv2d_19', channel = 49, transforms=True)


