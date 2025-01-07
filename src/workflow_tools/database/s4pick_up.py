import sys
import shutil
import random

# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from src.API_functions.Images import file_batch as fb
from pathlib import Path

def get_folder(input_path: Path, column_id: str=None) -> Path:
    """Get the folder based on the specified mode.
    
    Args:
        input_path (Path): Base input path
        column_id (str): Column ID (required for column_id mode)
    
    Returns:
        Path: Path to the harmonized folder
    """

    pattern = f"*/Soil.column.{column_id}"
    matching_folders = list(input_path.glob(pattern))
    
    if not matching_folders:
        print(f"Warning: No folder found for column {column_id}")
        return None
    
    return matching_folders[0] / "3.Harmonized"

def extract_images(input_folder: Path, config: dict, column_id: str = None):
    """Extract specified images and optional labels from the harmonized folder.

    Args:
        harmonized_folder (Path): Path to the harmonized folder
        config (dict): Configuration dictionary
        column_id (str, optional): Column ID for logging purposes
    """

    # ----------------- Input and output folders -----------------

    if config['mode'] == 'column_id':
        image_input = input_folder / "image"
        if not image_input.exists():
            raise FileNotFoundError(f"Image folder not found: {image_input}")
    else:
        image_input = input_folder

    image_output = config['image_output']

    if config.get('parallel_label', False):
        label_input = config['label_input']
        label_output = config['label_output']

    # Get a list of image files
    image_paths = fb.get_image_names(image_input, None, 'tif')
    images = [Path(image).name for image in image_paths]
    
    column_length = len(images)

    # ----------------- Image extraction -----------------
    
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

    # ----------------- Copy images and labels -----------------

    for idx in indices:
        if idx < column_length:
            img_name = images[idx]
            shutil.copyfile(image_input / img_name, image_output / img_name)
            if config.get('parallel_label', False):
                shutil.copyfile(label_input / img_name, label_output / img_name)

    return len(indices)



def validate_configuration(config: dict) -> bool:
    """Validate all required folders exist and are accessible. Leave image input folders unchecked, only the parent folder is checked.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if all validations pass, False otherwise
    """
    try:
        # input path
        input_path = config['base_input']
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return False
            
        # image output path
        output_path = config['output_folder']
        image_output = output_path / "image"
        image_output.mkdir(parents=True, exist_ok=True)
        config['image_output'] = image_output
        
        if config.get('parallel_label', False):
            # label input
            if config['mode'] != 'column_id':
                label_input = input_path / 'label'
                if not label_input.exists():
                    print(f"Error: Label folder not found: {label_input}")
                    return False
                else:
                    config['label_input'] = label_input
            
            # label output
            label_output = output_path / "label"
            label_output.mkdir(parents=True, exist_ok=True)
            config['label_output'] = label_output
            
        return True
        
    except Exception as e:
        print(f"Error during configuration validation: {str(e)}")
        return False

def process_all_columns(config: dict):
    """Process multiple soil columns or a single folder according to the specified configuration."""
    
    if not validate_configuration(config):
        print("Configuration validation failed. Exiting...")
        return
        
    input_path = config['base_input']
    log_dict = [] 

    # Process each column or only 1 folder
    if config['mode'] == 'column_id':
        for column_id in config['column_ids']:
            folder_to_be_proceed = get_folder(input_path, column_id=column_id)
            indices_nums = extract_images(folder_to_be_proceed, config, column_id)
            log_dict.append([indices_nums, column_id])
    else:
        folder_to_be_proceed = input_path
        indices_nums = extract_images(folder_to_be_proceed, config)
        log_dict.append([indices_nums, None])

    # Log the number of images extracted
    for indices_nums, column_id in log_dict:
        print(f"Extracted {indices_nums} images from column {column_id}")
        if config.get('parallel_label', False):
            print(f"Extracted {indices_nums} labels from column {column_id}")


if __name__ == "__main__":
    """
    Args:
        - base_input (str): Base input path. 
            - If 'column_id' mode, and 'parallel_label' mode is on, the images and labels are in 2 subfolders
            - Elif 'direct_folder' mode, there will be no label folder, the images are directly in the input folder, no subfolder.
        - output_folder (str): Output folder path
            - there will always be subfolder. 'image' and 'label' if 'parallel_label' mode is on.
        - mode (str): 'column_id' or 'direct_folder'
        - column_ids (list): List of column IDs

        - extraction_mode (str): 'random' or 'average'.
            - 'Average' for example: choose 1, 3, 5, ignore 2, 4, 6
        - continuous (bool): True to extract continuous images
        - images_per_section (int): Number of images to extract per section
        - num_sections (int): Number of sections to extract
        - random_seed (int): Random seed for reproducibility

        - parallel_label (bool): True to extract labels in parallel with.
    """
    config = {
        'base_input': Path(r'/mnt/g/DL_Data_raw/version7-large-lowRH/3.Harmonized/image'),
        'output_folder': Path(r'/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/temp'),
        'mode': 'direct_folder',  
        'column_ids': None,

        'extraction_mode': 'random',
        'continuous': False,
        'images_per_section': 1,
        'num_sections': 1,
        'random_seed': 8,

        'parallel_label': False,
    }
    
    process_all_columns(config)