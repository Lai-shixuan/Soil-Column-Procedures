#%%
# This file is a step following the raw2picturs
# 2 main functions: cut_roi and preprocess

#%%
import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from API_functions.Soils import pre_process
from API_functions import file_batch as fb
import pandas as pd
import cv2 as cv
import os
from tqdm import tqdm

#%%
# Using pandas to read a csv, each line stands for a roi region, and the columns are the starting point of roi and the size of roi

def get_roi_region_list(csv_file) -> list:
    roi_region_list = []

    # try to locate the id 28 and get its information
    for i in [28, 29, 30, 31, 32, 33, 34]:
        roi_x = int(csv_file.loc[csv_file['id'] == i, 'Cutted_starting_point_x'].values[0])
        roi_y = int(csv_file.loc[csv_file['id'] == i, 'Cutted_starting_point_y'].values[0])
        cutted_height_start = int(csv_file.loc[csv_file['id'] == i, 'Reversed_cut_h[0]'].values[0])
        cutted_height_end = int(csv_file.loc[csv_file['id'] == i, 'Reversed_cut_h[1]'].values[0])

        width = 320
        height = width

        roi_region_list.append(fb.roi_region(roi_x, roi_y, width, height, cutted_height_start, cutted_height_end))

    return roi_region_list

csv_file = pd.read_csv("f:/3.Experimental_Data/Soils/Metadata_of_whole_database.csv")
roi_region_list = get_roi_region_list(csv_file)

#%%
# cut_roi function

for i in [28, 29, 30, 31, 32, 33, 34]:
    fb.roi_select(path_in='f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.00' + str(i) + '/16bits/1.Reconstruct/', path_out='f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.00' + str(i) + '/16bits/2.ROI/', name_read=None, roi=roi_region_list[i-28], img_format='png')
    print('')
    print('')

#%%
# preprocess function


for i in [28, 29, 30, 31, 32, 33, 34]:

    path_in = 'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.00' + str(i) + '/16bits/2.ROI/'
    image_lists = fb.get_image_names(path_in, None, 'png')
    image_names = [os.path.basename(item) for item in image_lists]
    images = fb.read_images(image_lists, gray='gray', read_all=True)
    for j, image in enumerate(tqdm(images)):
        image = pre_process.map_range_to_65536(image, 30000, 35000)
        for k in range(1, 6, 1):
            for m in range(1, 10, 2):
                image = pre_process.noise_reduction(image, k, 500, 9)
                image = pre_process.high_boost_filter(image, 1, 1 + m/10)
                path_out='f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.00' + str(i) + '/16bits/3.Preprocess/noise' + str(k) + 'high_boost' + str(m) + '/'
                if not os.path.exists(path_out):
                    os.makedirs(path_out)
                cv.imwrite(os.path.join(path_out, image_names[j]), image)
    break