#%%
# Importing the libraries

import numpy as np
import cv2 as cv
import os
import sys

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from tqdm import tqdm
from API_functions import file_batch as fb
from API_functions.Soils import threshold_position_independent as tmi


#%%
# preprocess function

# for i in [28, 29, 30, 31, 32, 33, 34]:
path_in = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/3.Precheck/images/'
path_out='f:/3.Experimental_Data/Soils/Online/Soil.column.0035/4.Preprocess/remap/'
fb.windows_adjustment(path_in, path_out, min=None, max=None)


#%%
# Thresholding the images, to produce the label, if needed
# !!! do not run this cell if the images are already thresholded

for i in [28, 29, 30, 31, 32, 33, 34]:
    path_in = 'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.00' + str(i) + '/3.Preprocess/Remap/'
    image_lists = fb.get_image_names(path_in, None, 'png')
    image_names = [os.path.basename(item) for item in image_lists]
    images = fb.read_images(image_lists, gray='gray', read_all=True)
    images = np.array(images)

    for j, image in enumerate(tqdm(images)):
        image = tmi.otsu(image)
        path_out='f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.00' + str(i) + '/4.Threshold/Remap-otsu/'
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        cv.imwrite(os.path.join(path_out, image_names[j]), image)
    break