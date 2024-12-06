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

def extract_longitudinal_sections(path_in: str, col_id: str, label_square: tuple = None):
    """Extracts vertical sections and returns them instead of saving.

    Args:
        path_in: Input directory containing ROI-processed images
        col_id: Column ID for naming the output files
        label_square: Tuple of (width, depth) dimensions to calculate stretch ratio

    Returns:
        dict: Dictionary containing all section images
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
        
        return {
            'YZ_repeated': width_section_repeated,
            'XZ_repeated': height_section_repeated,
            'YZ_resized': width_section_resized,
            'XZ_resized': height_section_resized
        }
    else:
        return {
            'YZ': width_section,
            'XZ': height_section
        }

def combine_sections(images_list, spacing=50, target_width=210):
    """Combines multiple sections horizontally with spacing, handling different heights.
    
    Args:
        images_list: List of image arrays to combine
        spacing: Number of blank pixels between images
        target_width: Target width for all images after resizing
    """
    # Resize all images to same width while maintaining aspect ratio
    resized_images = []
    for img in images_list:
        scale = target_width / img.shape[1]
        new_height = int(img.shape[0] * scale)
        resized = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
        resized_images.append(resized)
    
    # Find maximum height after resizing
    max_height = max(img.shape[0] for img in resized_images)
    
    # Calculate total width needed
    total_width = (target_width * len(resized_images)) + (spacing * (len(resized_images) - 1))
    
    # Create blank canvas
    combined = np.full((max_height, total_width), 0, dtype=np.float32)
    
    # Place each resized image at the top of the canvas
    current_x = 0
    for img in resized_images:
        combined[0:img.shape[0], current_x:current_x + target_width] = img
        current_x += target_width + spacing
    
    return combined

def generate_summaries(sections_dict, output_dir):
    """Generates summary images from collected sections.
    
    Args:
        sections_dict: Dictionary of section types containing lists of images
        output_dir: Directory to save summary images
    """
    for section_type, images in sections_dict.items():
        if images:
            combined = combine_sections(images)
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

    # Collect all sections in dictionaries
    sections_dict = {
        'YZ_repeated': [],
        'XZ_repeated': [],
        'YZ_resized': [],
        'XZ_resized': []
    }

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
        
        # Get sections for this column
        column_sections = extract_longitudinal_sections(path_in, col_id, label_square=label_square)
        
        # Save individual sections and add to collections
        for section_type, image in column_sections.items():
            # Save individual image
            output_path = os.path.join(config['output_dir'], f'longitudinal_{section_type}_{col_id}.tif')
            cv2.imwrite(output_path, image)
            
            # Add to collection for summary
            sections_dict[section_type].append(image)

    # Generate summary images
    print('\nGenerating summary images...')
    generate_summaries(sections_dict, config['output_dir'])