import numpy as np
import keras
from glob import glob
import random
import os
import nibabel as nib

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size = 16, dim = (256, 256), channels = 1, seq = 't2'):
        'Initialization'

        self.path = path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = channels
        self.seq = seq
        self.subject_paths = glob(self.path + '*/*_'+self.seq+'.nii.gz')
        self.seg_paths = glob(self.path + '*/*_seg.nii.gz')
        self.mask_paths = glob(self.path + '*/mask.nii.gz')


    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.subject_paths)

    def _standardize_volume(self, volume, mask=None):
        """
                volume: volume which needs to be normalized
                mask: brain mask, only required if you prefer not to
                        consider the effect of air in normalization
        """
        if mask != None: volume = volume*mask

        mean = np.mean(volume[volume != 0])
        std = np.std(volume[volume != 0])
        
        return (volume - mean)/std


    def _normalize_volume(self, volume, mask=None, _type='MinMax'):
        """
                volume: volume which needs to be normalized
                mask: brain mask, only required if you prefer not to 
                        consider the effect of air in normalization
                _type: {'Max', 'MinMax', 'Sum'}
        """
        if mask != None: volume = mask*volume
        
        min_vol = np.min(volume)
        max_vol = np.max(volume)
        sum_vol = np.sum(volume)

        if _type == 'MinMax':
            return (volume - min_vol) / (max_vol - min_vol)
        elif _type == 'Max':
            return volume/max_vol
        elif _type == 'Sum':
            return volume/sum_vol
        else:
            raise ValueError("Invalid _type, allowed values are: {}".format('Max, MinMax, Sum'))
		
	
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        X, y = self.__data_generation()
        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples'

        # Initialization
        X = []
        y = []

        # Generate data
        while(len(X)) != self.batch_size:
            index_i = random.randint(0, len(self.subject_paths)-1)
            index_j = random.randint(20, 135)

            if os.path.exists(self.subject_paths[index_i]):

                # Store sample
                seq_vol = nib.load(self.subject_paths[index_i]).get_data()
                seg_vol = nib.load(self.seg_paths[index_i]).get_data()
                mask_vol = nib.load(self.mask_paths[index_i]).get_data()

                # preprocess volumes
                seq_vol = self._normalize_volume(self._standardize_volume(seq_vol*mask_vol))

                X.append(seq_vol[:, :, index_j][..., None])
                y.append(seg_vol[:, :, index_j][..., None])

        #print(np.array(y).shape, np.array(X).shape)

        return np.array(X), np.array(y)

