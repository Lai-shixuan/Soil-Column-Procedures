"""Module for extracting longitudinal sections from ROI-processed soil column images.

This module performs vertical sectioning of 3D soil column images to create two types of views:
1. Y-Z plane: A section through the middle of the column width
2. X-Z plane: A section through the middle of the column height

The module handles both pixel repetition and interpolation-based stretching to account for
physical dimensions of the soil columns.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from src.API_functions.Images import file_batch as fb
from src.API_functions.DL import multi_input_adapter as adapter

def extract_longitudinal_sections(path_in: str, col_id: str, label_square: tuple = None) -> tuple:
    """Extracts vertical sections from a stack of soil column images.

    Algorithm:
        1. Reads all images into a 3D numpy array (depth x height x width)
        2. Takes middle slices along width and height axes
        3. Normalizes images to float32 in range [0,1]
        4. Adjusts window level for better contrast
        5. If label_square provided:
           - Creates stretched versions using both pixel repetition and interpolation
           - Stretch ratio = label depth / label width

    Args:
        path_in (str): Directory containing ROI-processed images
        col_id (str): Column identifier for naming output files
        label_square (tuple, optional): (width, depth) dimensions for stretch calculation

    Returns:
        tuple: Contains:
            - dict: Sections dictionary with keys:
                - Without label_square: {'YZ': y-z_section, 'XZ': x-z_section}
                - With label_square: {'YZ_repeated', 'XZ_repeated', 'YZ_resized', 'XZ_resized'}
            - tuple: Original 3D stack shape
            - float: Stretch ratio used (1.0 if no label_square)
    """
    # Get image names and read all images
    images_paths = fb.get_image_names(path_in, None, 'png')
    img_stack = fb.read_images(images_paths, 'gray', read_all=True)
    img_stack = np.array(img_stack)
    
    original_shape = img_stack.shape

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
        
        return {
            'YZ_repeated': width_section_repeated,
            'XZ_repeated': height_section_repeated,
            'YZ_resized': width_section_resized,
            'XZ_resized': height_section_resized
        }, original_shape, ratio_decimal  # Added ratio_decimal to return
    else:
        return {
            'YZ': width_section,
            'XZ': height_section
        }, original_shape, 1.0

def combine_sections(images_list, original_heights, stretch_ratios, spacing=50, standard_height=3000):
    """Combines multiple section images into a single comparison image.

    Algorithm:
        1. Calculates height ratios based on original stack heights
        2. Resizes each image proportionally to maintain relative heights
        3. Creates a blank canvas with standard height
        4. Places each image sequentially with spacing
        5. Preserves aspect ratios during resizing

    Args:
        images_list (list): List of image arrays to combine
        original_heights (list): Original stack heights from 3D matrices
        stretch_ratios (list): Stretch ratios from label squares
        spacing (int, optional): Pixels between images. Defaults to 50.
        standard_height (int, optional): Target height for longest column. Defaults to 3000.

    Returns:
        ndarray: Combined image with all sections
    """
    # Calculate height ratios directly from original stack heights
    max_orig_height = max(original_heights)
    height_ratios = [h/max_orig_height for h in original_heights]
    
    # Calculate heights for each column based on standard_height and original ratios
    target_heights = [int(standard_height * ratio) for ratio in height_ratios]
    
    # Resize images maintaining width proportions but using target heights
    resized_images = []
    for img, target_height in zip(images_list, target_heights):
        # Calculate new width to maintain aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        new_width = int(target_height * aspect_ratio)
        resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized)
    
    # Calculate total width needed
    total_width = sum(img.shape[1] for img in resized_images) + (spacing * (len(resized_images) - 1))
    
    # Create blank canvas with standard height
    combined = np.full((standard_height, total_width), 0, dtype=np.float32)
    
    # Place each resized image at the top of the canvas
    current_x = 0
    for img in resized_images:
        combined[0:img.shape[0], current_x:current_x + img.shape[1]] = img
        current_x += img.shape[1] + spacing
    
    return combined

def generate_summaries(sections_dict, output_dir, original_heights, stretch_ratios):
    """Generates summary images for each type of section.

    Algorithm:
        1. Iterates through each section type (YZ, XZ, etc.)
        2. For each type, combines all column images into one summary
        3. Saves combined image as TIFF file

    Args:
        sections_dict (dict): Dictionary of section types containing lists of images
        output_dir (str): Directory to save summary images
        original_heights (list): Original stack heights for ratio calculation
        stretch_ratios (list): Stretch ratios from label squares

    Returns:
        None
    """
    for section_type, images in sections_dict.items():
        if images:
            # Use standard height of 3000 pixels
            combined = combine_sections(images, original_heights, stretch_ratios, 
                                     spacing=50, standard_height=3000)
            output_path = os.path.join(output_dir, f'summary_{section_type}.tif')
            cv2.imwrite(output_path, combined)
            print(f'Created summary for {section_type}')

if __name__ == '__main__':

    # Configuration
    config = {
        'base_path': "f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/",
        'output_dir': "f:/3.Experimental_Data/Soils/Dongying_Tiantan-Hospital/Analysis/vertical_sections/",
        'columns': {
            'start': 10,
            'end': 21
        },
        # Add label_square configurations for each column
        'label_squares': {
            'Soil.column.0010': (124023, 1000000),
            'Soil.column.0011': (72265.6, 950000),
            'Soil.column.0012': (118164, 980000),
            'Soil.column.0013': (131836, 975000),
            'Soil.column.0014': (139648, 990000),
            'Soil.column.0015': (129883, 995000),
            'Soil.column.0016': (124023, 985000),
            'Soil.column.0017': (72265.6, 970000),
            'Soil.column.0018': (118164, 960000),
            'Soil.column.0019': (131836, 965000),
            'Soil.column.0020': (139648, 975000),
            'Soil.column.0021': (129883, 980000),
        }
    }

    # Create output directory if it doesn't exist
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    # Collect all sections and original heights in dictionaries
    sections_dict = {
        'YZ_repeated': [],
        'XZ_repeated': [],
        'YZ_resized': [],
        'XZ_resized': []
    }
    original_heights = []
    stretch_ratios = []

    # Process each column and collect images
    for i in range(config['columns']['start'], config['columns']['end'] + 1):
        col_id = f'Soil.column.{i:04d}'
        path_in = os.path.join(config['base_path'], col_id, '2.ROI/')
        print(f'Processing longitudinal sections for {col_id}')
        
        # Get label_square for this specific column
        label_square = config['label_squares'].get(col_id)
        if not label_square:
            print(f'Warning: No label_square configuration found for {col_id}')
            continue
        
        # Get sections, original shape, and stretch ratio for this column
        column_sections, original_shape, stretch_ratio = extract_longitudinal_sections(
            path_in, col_id, label_square=label_square)
        original_heights.append(original_shape[0])
        stretch_ratios.append(stretch_ratio)
        
        # Save individual sections and add to collections
        for section_type, image in column_sections.items():
            # Save individual image
            output_path = os.path.join(config['output_dir'], f'longitudinal_{section_type}_{col_id}.tif')
            cv2.imwrite(output_path, image)
            
            # Add to collection for summary
            sections_dict[section_type].append(image)

    # Generate summary images with both ratios
    print('\nGenerating summary images...')
    generate_summaries(sections_dict, config['output_dir'], original_heights, stretch_ratios)