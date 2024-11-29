#%%
import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")
from API_functions import file_batch as fb

import os
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np


#%%
# Get the list of image files in the folder

path = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/0.Origin/Origin-normal_images_838/'
jpg_images = fb.get_image_names(path, None, 'jpg')
png_images = fb.get_image_names(path, None, 'png')

file_lists = jpg_images + png_images


#%%
# Divide the images into 2 groups based on the the number of white and black pixels

def divide_images_into_groups(image_paths):
    group_1 = []  # Larger images
    group_2 = []  # Smaller images
    
    # Assuming images are paired in the list
    for i in range(0, len(image_paths), 2):
        image_1 = image_paths[i]
        image_2 = image_paths[i+1]

        image_1_value = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE) 
        image_2_value = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)

        count_0_or_1 = np.sum((image_1_value == 0) | (image_1_value == 255))
        total_pixels = image_1_value.size
        ratio_1 = count_0_or_1 / total_pixels 

        count_0_or_1 = np.sum((image_2_value == 0) | (image_2_value == 255))
        total_pixels = image_2_value.size
        ratio_2 = count_0_or_1 / total_pixels
        
        # Compare the sizes and assign to groups
        if ratio_1 > 0.55:
            group_1.append(image_2)     
            group_2.append(image_1)     # labels group
        elif ratio_2 > 0.55:
            group_1.append(image_1)
            group_2.append(image_2)     # labels group
        else:
            print(f'Error: {image_1} and {image_2} are not paired images')
    
    return group_1, group_2

group_1, group_2 = divide_images_into_groups(file_lists)


#%%
# Change the names, extension, and save the names to a csv file

original_image_names = []
new_image_names = []
original_label_names = []
new_label_names = []

# Process image files
for i, image in enumerate(group_1):
    original_image_names.append(os.path.basename(image))
    new_name = f'0035.{i:03}.png'
    new_image_names.append(new_name)

# Process label files
for i, label in enumerate(group_2):
    original_label_names.append(os.path.basename(label))
    new_name = f'0035.{i:03}.png'
    new_label_names.append(new_name)

# Make 4 lists into 1 dict
name_dict = {
    'original_image_names': original_image_names,
    'new_image_names': new_image_names,
    'original_label_names': original_label_names,
    'new_label_names': new_label_names
}

# Save the dict to a csv file
name_df = pd.DataFrame(name_dict)
name_df.to_csv(path + 'name_dict.csv', index=False)


#%%
# Save the images to the new folders, with new names

image_paths = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/images/'
label_paths = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/labels/'


for i, img in enumerate(tqdm(group_1)):
    cv2.imwrite(image_paths + name_dict['new_image_names'][i], cv2.imread(img, cv2.IMREAD_GRAYSCALE))

for i, img in enumerate(tqdm(group_2)):
    cv2.imwrite(label_paths + name_dict['new_label_names'][i], cv2.imread(img, cv2.IMREAD_GRAYSCALE))

