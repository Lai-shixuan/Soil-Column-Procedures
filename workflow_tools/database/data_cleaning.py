#%%
import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")
from API_functions import file_batch as fb

import os
from tqdm import tqdm
import cv2
import pandas as pd


#%%
# Get the list of image files in the folder

path = 'f:/3.Experimental_Data/Soils/Online/'
jpg_images = fb.get_image_names(path, None, 'jpg')
png_images = fb.get_image_names(path, None, 'png')

file_lists = jpg_images + png_images


#%%
# Divide the images into 2 groups based on their sizes

def divide_images_into_groups(image_paths):
    group_1 = []  # Larger images
    group_2 = []  # Smaller images
    
    # Assuming images are paired in the list
    for i in range(0, len(image_paths), 2):
        image_1 = image_paths[i]
        image_2 = image_paths[i+1]
        
        # Get the sizes of the images
        size_1 = os.path.getsize(image_1)
        size_2 = os.path.getsize(image_2)
        
        # Compare the sizes and assign to groups
        if size_1 > size_2:
            group_1.append(image_1)
            group_2.append(image_2)
        else:
            group_1.append(image_2)
            group_2.append(image_1)
    
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
    new_name = f'{i:03}.png'
    new_image_names.append(new_name)

# Process label files
for i, label in enumerate(group_2):
    original_label_names.append(os.path.basename(label))
    new_name = f'{i:03}.png'
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

image_paths = path + 'images/'
label_paths = path + 'labels/'


for i, img in enumerate(tqdm(group_1)):
    cv2.imwrite(image_paths + name_dict['new_image_names'][i], cv2.imread(img))

for i, img in enumerate(tqdm(group_2)):
    cv2.imwrite(label_paths + name_dict['new_label_names'][i], cv2.imread(img))

