# %%
# This file is going to take 3d raw images and convert them to 2d images with preprocessing.
# It will saved in 16bit unsigned png format.

# %%
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from src.API_functions.Images import read_raw
from tqdm import tqdm

# %%
# Get the file names and their dimensions

path = "f:/3.Experimental_Data/Soils/Dongying_normal/Straightened/"
files = [f for f in os.listdir(path) if f.endswith('.raw')]
file_dict = {}
for file in files:
    parts = file.split('.')
    file_dict[parts[2]] = (
        int(parts[4]),      # width
        int(parts[5]),      # height
        int(parts[6]),      # depth
        file)               # File name

# %%
# There are 2 things to note, change if necessary:
# 1. The images are reversed in order, because avizo shows the images in reverse order, so many soil columns actually be reversed when saved
# 2. The images are converted from 16bit signed to 16bit unsigned

output_path = "f:/3.Experimental_Data/Soils/Dongying_normal/"

for key, pair in file_dict.items():
    if key != '0025':
        continue

    image3d = read_raw.read_raw(file_path=path + pair[3], height=pair[0], width=pair[1], depth=pair[2], dtype=np.dtype('<u2'),virtual=False)
    
    img_path = f'{output_path}/Soil.column.{key}/1.Reconstruct/'

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for i in tqdm(range(image3d.shape[0])):
        # Reverse the order of the images

        reversed_name = f'{image3d.shape[0] - i - 1:05d}'
        if key == '0026':
            reversed_name = f'{i:05d}'

        img_name = f'{key}-{reversed_name}-reconstruct.png' # {i:05d} or {reversed_name}
        img_full_path = os.path.join(img_path, img_name)

        # Convert to 16bit unsigned integer from 16bit signed
        # img_uint16 = (image3d[i] + 32768).astype(np.uint16)

        cv2.imwrite(img_full_path, image3d[i])  # image3d[i] or img_uint16
