#%%

import numpy as np
import cv2 as cv
import os
import sys

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from tqdm import tqdm
from pathlib import Path
from src.API_functions.Images import file_batch as fb
from src.API_functions.Soils import threshold_position_independent as tmi

#%%
# Adjust the windows of the images

for i in range(10, 22):
    path_in = f'f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/Soil.column.{i:04d}/3.Precheck/images/'
    path_out = f'f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/Soil.column.{i:04d}/4.Preprocess/Remap/'

    path_out = Path(path_out)
    if not path_out.exists():
        path_out.mkdir(parents=True)

    fb.windows_adjustment(path_in, path_out, min=None, max=None)


#%%
# Thresholding the images, to produce the label, if needed
# !!! do not run this cell if the images are already thresholded

for i in [28, 29, 30, 31, 32, 33, 34]:
    path_in = f'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.{i:04d}/4.Preprocess/Remap/'
    image_lists = fb.get_image_names(path_in, None, 'tif')
    image_names = [os.path.basename(item) for item in image_lists]
    images = fb.read_images(image_lists, gray='gray', read_all=True)
    images = np.array(images)

    for j, image in enumerate(tqdm(images)):
        mask = (image > 0) & (image < 1)
        
        # Pass both the original image and mask to kmeans_3d
        image = tmi.kmeans_3d(image, mask)
        # image = 1 - image
        
        path_out=f'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.{i:04d}/4.Threshold/Remap-kmeans/'
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        cv.imwrite(os.path.join(path_out, image_names[j]), image)
    break