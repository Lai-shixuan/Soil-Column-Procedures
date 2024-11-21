# %%
# This file is going to take 3d raw images and convert them to 2d images with preprocessing.
# It will saved in 16bit unsigned png format.

# %%
import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from API_functions import read_raw
import os
import cv2
import numpy as np

# %%
path = "f:/3.Experimental_Data/Soils/Quzhou_Henan/Straightened/"
files = [f for f in os.listdir(path) if f.endswith('.raw')]
file_dict = {}
for file in files:
    parts = file.split('.')
    file_dict[parts[2]] = (int(parts[4]), int(parts[5]), int(parts[6]), file)

# %%

output_path = "f:/3.Experimental_Data/Soils/Quzhou_Henan/"

for key, pair in file_dict.items():
    image3d = read_raw.read_raw(path + pair[3], pair[0], pair[1], pair[2], virtual=False)

    for i in range(image3d.shape[0]):
        reversed_name = f'{image3d.shape[0] - i - 1:03d}'
        img_name = output_path + 'Soil.column.' + key + '/16bits/1.Reconstruct/' + key + '.' + reversed_name + '.png'
        img_uint16 = (image3d[i] + 32768).astype(np.uint16)
        cv2.imwrite(img_name, img_uint16)