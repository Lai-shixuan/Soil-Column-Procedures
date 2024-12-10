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
from tqdm import tqdm


def extract_longitudinal_sections(path_in: str, label_square: tuple = None) -> tuple:
    """Extracts vertical sections from a stack of soil column images.

    Processes images one by one to reduce memory usage. Takes middle slices along width
    and height axes to create YZ and XZ plane sections.

    Args:
        path_in (str): Directory path containing the image stack.
        label_square (tuple, optional): Physical dimensions (width, depth) for stretch calculation.
            Used to adjust aspect ratio. Defaults to None.

    Returns:
        tuple: Contains three elements:
            - dict: Sections dictionary with keys depending on label_square:
                Without label_square: {'YZ': y-z_section, 'XZ': x-z_section}
                With label_square: {'YZ_repeated', 'XZ_repeated', 'YZ_resized', 'XZ_resized'}
            - tuple: Original 3D stack shape (depth, height, width)
            - float: Stretch ratio used (1.0 if no label_square)

    Raises:
        ValueError: If no images found in path_in.
    """
    # Get image names
    images_paths = fb.get_image_names(path_in, None, 'tif')
    
    # Read first image to get dimensions
    first_img = cv2.imread(images_paths[0], cv2.IMREAD_UNCHANGED)
    height, width = first_img.shape
    depth = len(images_paths)
    
    # Initialize arrays for sections
    mid_width = width // 2
    mid_height = height // 2
    width_section = np.zeros((depth, height), dtype=np.float32)    # YZ plane
    height_section = np.zeros((depth, width), dtype=np.float32)    # XZ plane
    
    # Process images one by one
    for z, img_path in enumerate(tqdm(images_paths)):
        # Read single image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Extract middle lines and store them
        width_section[z, :] = img[:, mid_width]    # YZ plane
        height_section[z, :] = img[mid_height, :]  # XZ plane
    
    # If label_square is provided, perform pixel repetition and interpolation
    if label_square is not None:
        # Calculate ratios
        ratio_int = int(np.ceil(label_square[1] / label_square[0]))
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
        }, (depth, height, width), ratio_decimal
    else:
        return {
            'YZ': width_section,
            'XZ': height_section
        }, (depth, height, width), 1.0

def combine_sections(images_list, original_heights, spacing=50, standard_height=3000):
    """Combines multiple section images into a single comparison image.

    Creates a composite image by placing all sections side by side while maintaining
    their relative height proportions.

    Args:
        images_list (list): List of image arrays to combine.
        original_heights (list): Original stack heights from 3D matrices.
        stretch_ratios (list): Stretch ratios calculated from label squares.
        spacing (int, optional): Pixels between images. Defaults to 50.
        standard_height (int, optional): Target height for longest column. Defaults to 3000.

    Returns:
        ndarray: Combined image array with all sections placed side by side.
            Shape is (standard_height, total_width) where total_width depends on
            input image widths and spacing.

    Note:
        Images are resized proportionally based on their original heights relative
        to the maximum height in the set.
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
    
    # Calculate minimum value from all sections instead of mean
    min_value = min(np.min(img) for img in resized_images)
    
    # Create blank canvas with minimum value instead of mean gray
    combined = np.full((standard_height, total_width), min_value, dtype=np.float32)
    
    # Place each resized image at the top of the canvas
    current_x = 0
    for img in resized_images:
        combined[0:img.shape[0], current_x:current_x + img.shape[1]] = img
        current_x += img.shape[1] + spacing
    
    return combined

def generate_summaries(sections_dict, output_dir, original_heights):
    """Generates summary images for each type of section.

    Creates and saves combined summary images for each section type (YZ, XZ)
    by combining all columns into single comparison images.

    Args:
        sections_dict (dict): Dictionary of section types containing lists of images.
            Expected keys are combinations of 'YZ'/'XZ' with '_repeated'/'_resized'.
        output_dir (str): Directory path where summary images will be saved.
        original_heights (list): Original stack heights for ratio calculation.
        stretch_ratios (list): Stretch ratios from label squares.

    Returns:
        None

    Note:
        Saves TIFF files named 'summary_{section_type}.tif' in output_dir.
        Prints confirmation message for each summary created.
    """
    for section_type, images in sections_dict.items():
        if images:
            # Use standard height of 3000 pixels
            combined = combine_sections(images, original_heights, spacing=50, standard_height=3000)

            output_path = os.path.join(output_dir, f'summary_{section_type}.tif')
            cv2.imwrite(output_path, combined)
            print(f'Created summary for {section_type}')

if __name__ == '__main__':
    # Configuration
    config = {
        'base_path': "f:/3.Experimental_Data/Soils/Dongying_normal/",
        'output_dir': "f:/3.Experimental_Data/Soils/Dongying_normal/Analysis/vertical_sections/",
        'columns': [i for i in range(22, 28)],  # Specific column numbers to process
        'label_squares': {
            'Soil.column.0022': (1, 1),
            'Soil.column.0023': (1, 1),
            'Soil.column.0024': (1, 1),
            'Soil.column.0025': (1, 1),
            'Soil.column.0026': (1, 1),
            'Soil.column.0027': (1, 1),
            # 'Soil.column.0028': (78125, 1000000),
            # 'Soil.column.0029': (78125, 1000000),
            # 'Soil.column.0030': (78125, 1000000),
            # 'Soil.column.0031': (78125, 1000000),
            # 'Soil.column.0032': (78125, 1000000),
            # 'Soil.column.0033': (78125, 1000000),
            # 'Soil.column.0034': (78125, 1000000)

            # 'Soil.column.0010': (124023, 1000000),
            # 'Soil.column.0011': (72265.6, 1000000),
            # 'Soil.column.0012': (118164, 1000000),
            # 'Soil.column.0013': (131836, 1000000),
            # 'Soil.column.0014': (139648, 1000000),
            # 'Soil.column.0015': (129883, 1000000),

            # 'Soil.column.0016': (124023, 1000000),
            # 'Soil.column.0017': (72265.6, 1000000),
            # 'Soil.column.0018': (118164, 1000000),
            # 'Soil.column.0019': (131836, 1000000),
            # 'Soil.column.0020': (139648, 1000000),
            # 'Soil.column.0021': (129883, 1000000),
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

    # Process specific columns
    for col_num in config['columns']:
        col_id = f'Soil.column.{col_num:04d}'
        path_in = os.path.join(config['base_path'], col_id, '3.Harmonized/image')
        print(f'Processing longitudinal sections for {col_id}')
        
        # Get label_square for this specific column
        label_square = config['label_squares'].get(col_id)
        if not label_square:
            raise ValueError(f'Label square not found for {col_id}')
        
        # Get sections, original shape, and stretch ratio for this column
        column_sections, original_shape, stretch_ratio = extract_longitudinal_sections(path_in, label_square)
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
    generate_summaries(sections_dict, config['output_dir'], original_heights)