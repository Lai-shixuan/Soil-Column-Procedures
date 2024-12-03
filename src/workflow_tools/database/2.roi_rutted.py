"""Module for handling ROI (Region of Interest) selection in soil column images.

This module provides functionality to extract ROI information from CSV files
and apply ROI selection to a batch of images. It is specifically designed to work
with soil column images from the Quzhou Henan dataset.

Typical usage example:
    csv_file = pd.read_csv("path/to/metadata.csv")
    roi_region_list = get_roi_region_list(csv_file)
    fb.roi_select(path_in, path_out, roi=roi_region_list[0])
"""

import pandas as pd
import sys

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from src.API_functions.Images import file_batch as fb


def get_roi_region_list(csv_file) -> list:
    """Extracts ROI region inforation for specific soil columns from a CSV file.

    Args:
        csv_file: A pandas DataFrame containing ROI metadata for soil columns.
            Expected columns include 'id', 'Cutted_starting_point_x',
            'Cutted_starting_point_y', 'Reversed_cut_h[0]', and 'Reversed_cut_h[1]'.

    Returns:
        list: A list of fb.roi_region objects containing ROI parameters for soil 
            columns with IDs 28-34. Each roi_region object contains:
            - x, y coordinates of the ROI
            - width and height (both set to 320)
            - cutted height start and end points

    """
    roi_region_list = []
    
    # Define column names as variables
    x_col = 'Cutted_starting_point_x'
    y_col = 'Cutted_starting_point_y'
    h_start_col = 'Reversed_cut_h[0]'
    h_end_col = 'Reversed_cut_h[1]'
    width_col = 'Cutted_width'
    id_col = 'id'

    # try to locate the id 28 and get its information
    for i in [28, 29, 30, 31, 32, 33, 34]:
        roi_x = int(csv_file.loc[csv_file[id_col] == i, x_col].values[0])
        roi_y = int(csv_file.loc[csv_file[id_col] == i, y_col].values[0])
        cutted_height_start = int(csv_file.loc[csv_file[id_col] == i, h_start_col].values[0])
        cutted_height_end = int(csv_file.loc[csv_file[id_col] == i, h_end_col].values[0])
        width = int(csv_file.loc[csv_file[id_col] == i, width_col].values[0])
        height = width

        roi_region_list.append(fb.roi_region(roi_x, roi_y, width, height, cutted_height_start, cutted_height_end))

    return roi_region_list


if __name__ == '__main__':
    csv_file = pd.read_csv("f:/3.Experimental_Data/Soils/Metadata_of_whole_database.csv")
    roi_region_list = get_roi_region_list(csv_file)

    for i in [28, 29, 30, 31, 32, 33, 34]:
        path_in = f'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.{i:04d}/1.Reconstruct/'
        path_out = f'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.{i:04d}/2.ROI/'
        fb.roi_select(path_in, path_out, name_read=None, roi=roi_region_list[i-28], img_format='png')
        print('')

        break