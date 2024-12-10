import sys
import shutil
import random

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from src.API_functions.Images import file_batch as fb
from pathlib import Path

def extract_images(column_id: str, config: dict):
    """Extract specified images from the input folder based on configured rules.

    Args:
        column_id (str): The ID of the column (e.g., '0028').
        config (dict): Configuration dictionary containing:
            - base_input (str): Base directory for input images
            - output_folder (str): Base directory for output images
            - extraction_mode (str): 'random' or 'average'
            - continuous (bool): Whether to extract continuous images
            - images_per_section (int): Number of images to extract per section
            - num_sections (int): Number of sections to divide the column into
            - random_seed (int): Random seed for reproducibility

    Returns:
        None
    """
    # Convert to Path objects from config
    input_folder = Path(config['base_input'])
    output_folder = Path(config['output_folder'])
    
    # Find the column folder by pattern matching
    pattern = f"*/Soil.column.{column_id}"
    matching_folders = list(input_folder.glob(pattern))
    
    if not matching_folders:
        print(f"Warning: No folder found for column {column_id}")
        return
    
    # Use the first matching folder (should be unique)
    column_folder = matching_folders[0] / "3.Harmonized" / "image"
    
    # Get a list of image files
    images = fb.get_image_names(column_folder, None, 'tif')
    images = [Path(image).name for image in images]
    
    column_length = len(images)
    
    if config['extraction_mode'] == 'random':
        random.seed(config['random_seed'])
        if config['continuous']:
            # Random start point for continuous extraction
            max_start = column_length - config['images_per_section']
            start_idx = random.randint(0, max_start)
            indices = list(range(start_idx, start_idx + config['images_per_section']))
        else:
            # Random indices for non-continuous extraction
            indices = random.sample(range(column_length), config['images_per_section'])
            indices.sort()
    
    else:  # 'average' mode
        section_size = column_length // config['num_sections']
        indices = []
        for section in range(config['num_sections']):
            section_start = section * section_size
            if config['continuous']:
                # Take continuous images from each section
                start_idx = section_start + (section_size - config['images_per_section']) // 2
                indices.extend(range(start_idx, start_idx + config['images_per_section']))
            else:
                # Take random images from each section
                section_indices = random.sample(
                    range(section_start, section_start + section_size),
                    config['images_per_section']
                )
                indices.extend(section_indices)

    # Create output folder if it doesn't exist
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Copy selected images directly to output folder, keeping original names
    for idx in indices:
        if idx < column_length:
            src_path = column_folder / images[idx]
            dst_path = output_folder / images[idx]
            shutil.copyfile(src_path, dst_path)

    print(f"Extracted {len(indices)} images from column {column_id} to {output_folder}")

def process_all_columns(config: dict):
    """Process multiple soil columns according to the specified configuration."""
    for column_id in config['column_ids']:
        extract_images(column_id, config)

if __name__ == "__main__":
    # Configuration dictionary
    config = {
        # Path configurations
        'base_input': "f:/3.Experimental_Data/Soils/",
        'output_folder': "g:/DL_Data_raw/version4-classes/",
        'column_ids': [f"{i:04d}" for i in range(9, 36)] + ['0003', '0005', '0007'],
        
        # Extraction configurations
        'extraction_mode': 'random',    # 'random' or 'average'
        'continuous': True,
        'images_per_section': 3,
        'num_sections': 1,

        # Seed
        'random_seed': 44,
    }
    
    process_all_columns(config)