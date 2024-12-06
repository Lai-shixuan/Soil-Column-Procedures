"""Module for extracting longitudinal sections from ROI-processed soil column images.

This module creates two vertical sections from each soil column image:
1. A section through the middle of the width (Y-Z plane)
2. A section through the middle of the height (X-Z plane)

"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from src.API_functions.Images import file_batch as fb
from src.API_functions.DL import multi_input_adapter as adapter

def extract_longitudinal_sections(path_in: str, path_out: str, label_square: tuple = None):
    """Extracts vertical sections from the middle of width and height of ROI images.

    Args:
        path_in: Input directory containing ROI-processed images
        path_out: Output directory for saving the longitudinal sections
        label_square: Tuple of (width, depth) dimensions to calculate stretch ratio
    """
    # Get image names and read all images
    images_paths = fb.get_image_names(path_in, None, 'png')
    img_stack = fb.read_images(images_paths, 'gray', read_all=True)
    img_stack = np.array(img_stack)

    # Get middle sections directly from 3D array
    mid_width = img_stack.shape[2] // 2
    mid_height = img_stack.shape[1] // 2
    
    width_section = img_stack[:, :, mid_width]    # YZ plane
    height_section = img_stack[:, mid_height, :]  # XZ plane

    # Change the images to float32
    width_section = adapter.harmonized_normalize(width_section)
    height_section = adapter.harmonized_normalize(height_section)

    # Adjust window level and width
    width_section = fb.windows_adjustment_one_image(width_section)
    height_section = fb.windows_adjustment_one_image(height_section)

    if label_square is not None:
        # Calculate integer ratio for pixel repetition
        ratio_int = int(np.ceil(label_square[1] / label_square[0]))
        
        # Calculate decimal ratio for resize (rounded to 1 decimal place)
        ratio_decimal = round(label_square[1] / label_square[0], 1)
        
        # Create repeated pixel versions
        width_section_repeated = np.repeat(width_section, ratio_int, axis=0)
        height_section_repeated = np.repeat(height_section, ratio_int, axis=0)
        
        # Create interpolated versions
        new_height = int(width_section.shape[0] * ratio_decimal)
        width_section_resized = cv2.resize(width_section, (width_section.shape[1], new_height))
        height_section_resized = cv2.resize(height_section, (height_section.shape[1], new_height))
        
        # Save all versions
        cv2.imwrite(os.path.join(path_out, 'longitudinal_YZ_repeated.tif'), width_section_repeated)
        cv2.imwrite(os.path.join(path_out, 'longitudinal_XZ_repeated.tif'), height_section_repeated)
        cv2.imwrite(os.path.join(path_out, 'longitudinal_YZ_resized.tif'), width_section_resized)
        cv2.imwrite(os.path.join(path_out, 'longitudinal_XZ_resized.tif'), height_section_resized)
    else:
        # Original behavior when label_square is None
        cv2.imwrite(os.path.join(path_out, 'longitudinal_YZ.tif'), width_section)
        cv2.imwrite(os.path.join(path_out, 'longitudinal_XZ.tif'), height_section)

if __name__ == '__main__':

    # Example usage with different image types
    for i in range(10, 22):
        path_in = f'f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/Soil.column.{i:04d}/2.ROI/'
        path_out = f'f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/Soil.column.{i:04d}/_Analysis/vertical_sects/'
        
        if not os.path.exists(path_out):
            os.makedirs(path_out)
            
        # Choose appropriate img_type based on your data
        extract_longitudinal_sections(path_in, path_out, label_square=(78125, 1000000))
        print(f'Processed longitudinal sections for soil column {i}')