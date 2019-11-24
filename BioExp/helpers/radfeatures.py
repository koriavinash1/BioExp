import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
import radiomics
from tqdm import tqdm
from radiomics import firstorder, glcm, imageoperations, glrlm, glszm, ngtdm, gldm, getTestCase

class ExtractRadiomicFeatures():
    def __init__(self, input_image, 
                    input_mask=None, 
                    save_path=None, 
                    seq='Flair',
                    class_ = 'ET',
                    all_ = True):
        
        self.img = input_image
        try:
            if not input_mask.any():
                self.GT = np.ones(tuple(list(self.img.shape)))
            else: self.GT = input_mask
        except:
            if not input_mask: self.GT = np.ones(tuple(list(self.img.shape)))
            else: self.GT = input_mask
        self.GT = sitk.GetImageFromArray(self.GT)
        self.img = sitk.GetImageFromArray(self.img) #, isVector=False)
        self.save_path = save_path
        self.seq = seq
        self.all_ = all_
        self.class_ = class_
        self.feat_dict = {}


    def first_order(self):

        feat_dict = {}
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(self.img, self.GT)
        firstOrderFeatures.enableAllFeatures()
        firstOrderFeatures.execute()          
        for (key,val) in six.iteritems(firstOrderFeatures.featureValues):
            print (key, val)
            self.feat_dict[key] = [val]
            feat_dict[key] = [val]

        print (feat_dict)
        df = pd.DataFrame(feat_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'firstorder_features.csv'), index=False)

        return df


    def glcm_features(self):

        glcm_dict = {}
        GLCMFeatures = glcm.RadiomicsGLCM(self.img, self.GT)
        GLCMFeatures.enableAllFeatures()
        GLCMFeatures.execute()
        for (key,val) in six.iteritems(GLCMFeatures.featureValues):
            self.feat_dict[key] = [val]
            glcm_dict[key] = [val]

        df = pd.DataFrame(glcm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'glcm_features.csv'), index=False)

        return df


    def glszm_features(self):
        
        glszm_dict = {}
        GLSZMFeatures = glszm.RadiomicsGLSZM(self.img, self.GT)
        GLSZMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLSZMFeatures.execute()
        for (key,val) in six.iteritems(GLSZMFeatures.featureValues):
            self.feat_dict[key] = [val]
            glszm_dict[key] = [val]
        print (glszm_dict)
        df = pd.DataFrame(glszm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'glszm_features.csv'), index=False)


        return df
    
    
    def glrlm_features(self):


        glrlm_dict = {}
        GLRLMFeatures = glrlm.RadiomicsGLRLM(self.img, self.GT)
        GLRLMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLRLMFeatures.execute()
        for (key,val) in six.iteritems(GLRLMFeatures.featureValues):
            self.feat_dict[key] = [val]
            glrlm_dict[key] = [val]
        print (glrlm_dict)
        df = pd.DataFrame(glrlm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'glrlm_features.csv'), index=False)

    
        return df
    
    
    def ngtdm_features(self):
        
        ngtdm_dict =  {}
        NGTDMFeatures = ngtdm.RadiomicsNGTDM(self.img, self.GT)
        NGTDMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        NGTDMFeatures.execute()
        for (key,val) in six.iteritems(NGTDMFeatures.featureValues):
            self.feat_dict[key] = [val]
            ngtdm_dict[key] = [val]

        print (ngtdm_dict)
        df = pd.DataFrame(ngtdm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'ngtdm_features.csv'), index=False)
    
        return df

    def gldm_features(self):

        gldm_dict = {}
        GLDMFeatures = gldm.RadiomicsGLDM(self.img, self.GT)
        GLDMFeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default.
        GLDMFeatures.execute()
        for (key,val) in six.iteritems(GLDMFeatures.featureValues):

            self.feat_dict[key] = [val]
            gldm_dict[key] = [val]

        print (gldm_dict)
        df = pd.DataFrame(gldm_dict)
        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'gldm_features.csv'), index=False)

        return df

    
    def all_features(self):

        _ = self.first_order()
        _ = self.glcm_features()
        _ = self.glszm_features()
        _ = self.glrlm_features()
        #_ = self.ngtdm_features()
        # _ = self.gldm_features()
        
        df = pd.DataFrame(self.feat_dict)

        if self.save_path:
            df.to_csv(os.path.join(self.save_path, 'all_features.csv'), index=False)

        return df

