import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob 
from PIL import Image

root_path = '/home/pi/Projects/BioExp/results/Dissection/SimNet'
paths = glob(root_path + '/*.png')

for i, img_path in enumerate(paths):
	img_id = img_path.split("/")[-1].split("_")[0]
	feature_id = img_path.split("/")[-1][3:].split(".")[0]
	img = Image.open(img_path)
	row_size, col_size = img.size[0]//6, img.size[1]//6
	root_path_ = os.path.join(root_path, feature_id)
	print (img_id)
	orig_img = img.crop((0, 0, row_size, col_size))
	
	for xi in range(0, 6):
		for yi in range(0, 6):
			save_path = os.path.join(root_path_, str((xi*6 + yi)))
			if not os.path.exists(save_path): 
				os.makedirs(save_path)	

			concept = img.crop((yi*row_size, xi*col_size, (yi+1)*row_size, (xi+1)*col_size))
			
			concept.save(os.path.join(save_path, img_id +'_' +str((xi*6 + yi))+ '.png'))
			
			orig_img.save(os.path.join(root_path_, img_id + '.png'))
