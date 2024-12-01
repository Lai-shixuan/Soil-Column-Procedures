#%%
# Importing necessary libraries

import cv2
import sys
import numpy as np
# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from API_functions import file_batch as fb
from API_functions.Soils import threshold_position_independent as tpi


#%%
# Define the function to cut the circle

def cut_circle(image: np.ndarray, r: int=512) -> np.ndarray:
    # Create a single-channel mask
    mask = np.zeros(image.shape, dtype=np.uint16)
    
    # Get image dimensions and calculate center
    height, width = image.shape
    center = (width // 2, height // 2)
    
    # Draw circle on mask (value 1 for the circle area)
    cv2.circle(mask, center, r, 1, -1)
    
    # Use multiply to preserve original pixel values
    result = cv2.multiply(image, mask)
    
    return result


#%%
# Batch cut the circle
pathin = Path('g:/DL_Data_raw/Unit_test/round_square_inference/images/')

# same folder for input and output, for inverse labels
# pathin = Path('g:/DL_Data_raw/Unit_test/round_square_inference/labels/')
pathout = Path('g:/DL_Data_raw/Unit_test/round_square_inference/images_cutted/')

images_paths = fb.get_image_names(pathin, None, 'png')
images = fb.read_images(images_paths, 'gray', read_all=True)

for i in range(len(images)):
    img = cut_circle(images[i], r=200)
    image_name = Path(images_paths[i]).stem + '.png'  # Add file extension
    # Save as 16-bit PNG
    cv2.imwrite(str(pathout / image_name), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Use path / operator and convert to string


#%%


pathin = Path('g:/DL_Data_raw/Unit_test/round_square_inference/images_cutted/')
pathout = Path('g:/DL_Data_raw/Unit_test/round_square_inference/labels/')

images_paths = fb.get_image_names(pathin, None, 'png')
images = fb.read_images(images_paths, 'gray', read_all=True)

for i, img in enumerate(images):
    img = tpi.user_threshold(img, 33390)
    # img = tpi.kmeans_3d(img)
    img = cv2.bitwise_not(img)
    image_name = Path(images_paths[i]).stem + '.png'  # Add file extension
    cv2.imwrite(str(pathout / image_name), img)


#%%


# pathin = Path('g:/DL_Data_raw/Unit_test/round_square_inference/images_cutted/')
pathout = Path('g:/DL_Data_raw/Unit_test/round_square_inference/labels/')
# pathout = Path('g:/DL_Data_raw/Unit_test/round_square_inference/labels/')

# fb.windows_adjustment(pathin, pathout, min=33000, max=34119)
fb.bitconverter.convert_to_8bit(pathout, pathout, 'png')