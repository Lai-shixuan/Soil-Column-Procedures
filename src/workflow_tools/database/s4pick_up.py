import sys
import shutil
import random

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from src.API_functions.Images import file_batch as fb
from pathlib import Path

def extract_images(column_id: str, config: dict):
    """Extract specified images and optional labels from the input folder.

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
            - parallel_label (bool): Whether to extract labels in parallel
            - label_folder (str): Name of the label subfolder (e.g., 'label')

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
    harmonized_folder = matching_folders[0] / "3.Harmonized"
    image_folder = harmonized_folder / "image"
    
    # Get a list of image files
    images = fb.get_image_names(image_folder, None, 'tif')
    images = [Path(image).name for image in images]
    
    column_length = len(images)
    
    if config['extraction_mode'] == 'random':
        random.seed(config['random_seed'])
        section_size = column_length // config['num_sections']
        indices = []
        
        for section in range(config['num_sections']):
            section_start = section * section_size
            section_end = section_start + section_size
            
            if config['continuous']:
                # Random start point for continuous extraction within section
                max_start = section_end - config['images_per_section']
                start_idx = random.randint(section_start, max_start)
                section_indices = list(range(start_idx, start_idx + config['images_per_section']))
            else:
                # Random indices within this section
                section_indices = random.sample(
                    range(section_start, section_end),
                    config['images_per_section']
                )
            indices.extend(sorted(section_indices))
    
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
                # Take images_per_section images from each section
                section_indices = random.sample(
                    range(section_start, section_start + section_size),
                    min(config['images_per_section'], section_size)  # Ensure we don't exceed section size
                )
                indices.extend(sorted(section_indices))

    # Create output folders
    image_output = output_folder / "image"
    image_output.mkdir(parents=True, exist_ok=True)

    if config.get('parallel_label', False):
        label_folder = harmonized_folder / 'label'
        label_output = output_folder / 'label'
        label_output.mkdir(parents=True, exist_ok=True)
        
        # Verify that labels exist for all selected images
        for img_name in images:
            if not (label_folder / img_name).exists():
                print(f"Warning: Label missing for {img_name} in column {column_id}")
                return

    # Copy selected files
    for idx in indices:
        if idx < column_length:
            img_name = images[idx]
            # Copy image
            src_path = image_folder / img_name
            dst_path = image_output / img_name
            shutil.copyfile(src_path, dst_path)

            # Copy corresponding label if enabled
            if config.get('parallel_label', False):
                label_src = label_folder / img_name
                label_dst = label_output / img_name
                shutil.copyfile(label_src, label_dst)

    print(f"Extracted {len(indices)} images from column {column_id} to {image_output}")
    if config.get('parallel_label', False):
        print(f"Extracted {len(indices)} labels from column {column_id} to {label_output}")

def process_all_columns(config: dict):
    """Process multiple soil columns according to the specified configuration."""
    for column_id in config['column_ids']:
        extract_images(column_id, config)

if __name__ == "__main__":
    # Configuration dictionary
    config = {
        # Path configurations
        'base_input': r"f:/3.Experimental_Data/Soils/",
        'output_folder': r"g:\DL_Data_raw\version7-large-lowRH\3.Harmonized",
        'column_ids': [f"{i:04d}" for i in range(28, 35)] + [f"{i:04d}" for i in range(16, 22)],  # [f"{i:04d}" for i in range(9, 36)] + ['0003', '0005', '0007'],
        
        # Extraction configurations
        'extraction_mode': 'random',    # 'random' or 'average'
        'continuous': False,
        'images_per_section': 10,
        'num_sections': 35,

        # Seed
        'random_seed': 48,

        # New parallel label configurations
        'parallel_label': False,
        'label_folder': 'label',  # subfolder name for labels
    }
    
    process_all_columns(config)