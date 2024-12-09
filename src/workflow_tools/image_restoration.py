import cv2
import pandas as pd
import sys

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from tqdm import tqdm
from src.API_functions.DL import multi_input_adapter
from src.API_functions.Images import file_batch as fb
from src.API_functions.DL.shape_detectors import EllipseParams, RectangleParams

def reconstruct_data_from_csv(csv_path: str, patches: list) -> dict:
    """
    Reconstruct the data structure needed for restoration from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing patch information
        patches: List of patch images
    Returns:
        dict: Data structure compatible with restore_image_batch
    """
    df = pd.read_csv(csv_path)
    unique_images = df['original_image_index'].unique()
    
    patch_positions = []
    original_image_info = []
    patch_to_image_map = df['original_image_index'].tolist()
    shape_params = []
    padding_info_list = []
    
    for img_idx in unique_images:
        img_patches = df[df['original_image_index'] == img_idx]
        # Get positions for this image
        positions = list(zip(img_patches['position_x'], img_patches['position_y']))
        patch_positions.append(positions)
        
        # Get original dimensions
        original_shape = (
            int(img_patches['original_height'].iloc[0]),
            int(img_patches['original_width'].iloc[0])
        )
        original_image_info.append(original_shape)
        
        # Reconstruct shape parameters based on shape_type
        first_patch = img_patches.iloc[0]
        shape_type = first_patch['shape_type']
        
        if shape_type == 'none':
            params = None
        elif shape_type == 'ellipse':
            params = EllipseParams(
                center=(first_patch['center_x'], first_patch['center_y']),
                covered_pixels=first_patch['covered_pixels'],
                long_axis=first_patch['long_axis'],
                short_axis=first_patch['short_axis']
            )
        else:  # rectangle
            params = RectangleParams(
                center=(first_patch['center_x'], first_patch['center_y']),
                covered_pixels=first_patch['covered_pixels'],
                width=first_patch['width'],
                height=first_patch['height']
            )
        shape_params.append(params)
    
    for i in range(len(patches)):
        patch_data = df.iloc[i]
        if pd.notna(patch_data['padding_top']):
            padding_info = {
                'top': int(patch_data['padding_top']),
                'bottom': int(patch_data['padding_bottom']),
                'left': int(patch_data['padding_left']),
                'right': int(patch_data['padding_right']),
                'color': int(patch_data['padding_color'])
            }
        else:
            padding_info = None
        padding_info_list.append(padding_info)
    
    return {
        'patches': patches,
        'patch_positions': patch_positions,
        'original_image_info': original_image_info,
        'patch_to_image_map': patch_to_image_map,
        'shape_params': shape_params,
        'padding_info': padding_info_list  # Add padding info to results
    }

def restore_images_workflow(input_images_path: str, output_dir: str, csv_path: str = None):
    """
    Workflow function to restore images from patches
    Args:
        input_images_path: Path to input images
        output_dir: Directory to save restored images
        csv_path: Optional path to CSV file containing patch information
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_paths = fb.get_image_names(input_images_path, None, 'tif')
    patches = fb.read_images(image_paths, 'gray', read_all=True)
    
    if csv_path:
        datasets = reconstruct_data_from_csv(csv_path, patches)
    
    restored_images = multi_input_adapter.restore_image_batch(datasets, target_size=512)
    
    # Group patches by their original image names
    if image_paths:
        original_names = []
        for path in image_paths:
            name = Path(path).stem  # e.g., "0035.000.circle_patch_0000"
            base_name = name.split('_patch_')[0]  # e.g., "0035.000.circle"
            if base_name not in original_names:
                original_names.append(base_name)
        
        # Save restored images with corresponding original names
        for img_idx, image in enumerate(tqdm(restored_images)):
            output_filename = f"{original_names[img_idx]}.restored.tif"
            output_path = str(Path(output_dir) / output_filename)
            cv2.imwrite(output_path, image)
    
    return restored_images

if __name__ == "__main__":
    """
    Direct execution configuration for VSCode
    """
    # Configuration
    config = {
        'input_images_path': 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/3.Precheck/labels/',
        'output_dir': 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/6.Restoration/labels/',
        'csv_path': 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/3.Precheck/metadata/label_patches.csv'
    }
    
    print(f"Processing patches from {config['input_images_path']}")
    restored = restore_images_workflow(**config)
    print(f"Successfully restored {len(restored)} images to {config['output_dir']}")