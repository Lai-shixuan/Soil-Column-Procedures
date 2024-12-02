#%%
# This file is a step following the raw2picturs
import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from API_functions import file_batch as fb
import pandas as pd


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
