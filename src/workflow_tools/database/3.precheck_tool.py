import cv2
import logging
import os
import glob
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from typing import List
from src.API_functions.Images import file_batch as fb
from src.API_functions.DL import multi_input_adapter

def batch_precheck_and_save(
    image_paths: List[str],
    output_dir: str,
    is_label: bool = False
) -> dict:
    """
    Perform precheck on a batch of images and save the resulting patches
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save the output patches
        is_label: Whether the images are labels (default: False)
    
    Returns:
        dict: The precheck results containing patch information
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directories
    patch_dir = os.path.join(output_dir, 'labels' if is_label else 'images')
    Path(patch_dir).mkdir(parents=True, exist_ok=True)

    # Read images
    images = fb.read_images(image_paths, 'gray', read_all=True)

    # Perform precheck
    try:
        results = multi_input_adapter.precheck(images, is_label=is_label)
    except Exception as e:
        logger.error(f"Precheck failed: {str(e)}")
        raise

    # Save patches with original filenames
    patch_to_image_map = results['patch_to_image_map']
    for i, img in enumerate(results['patches']):
        # Get original image path and extract filename
        original_path = image_paths[patch_to_image_map[i]]
        original_filename = os.path.splitext(os.path.basename(original_path))[0]
        # Create new filename with original name and patch number
        output_path = os.path.join(patch_dir, f'{original_filename}_patch_{i:04d}.tif')
        cv2.imwrite(output_path, img)

    # Log information
    logger.info("Precheck Results:")
    logger.info(f"Patch positions: {results['patch_positions'][:3]}")
    logger.info(f"Original image info: {results['original_image_info'][:3]}")
    logger.info(f"Patch to image map: {results['patch_to_image_map'][:3]}")
    logger.info(f"Shape parameters: {results['shape_params'][:3]}")

    return results

if __name__ == "__main__":
    
    # Example paths - modify these according to your data location
    image_dir = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/2.ROI/image/'
    label_dir = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/2.ROI/label/'
    output_dir = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/3.Precheck/'
    
    # Get all image files
    image_paths = glob.glob(f"{image_dir}/*.tif")  # adjust file extension as needed
    label_paths = glob.glob(f"{label_dir}/*.tif")
    
    # Process regular images
    print("Processing images...")
    image_results = batch_precheck_and_save(
        image_paths=image_paths,
        output_dir=output_dir,
        is_label=False
    )
    
    # Process label images
    print("Processing labels...")
    label_results = batch_precheck_and_save(
        image_paths=label_paths,
        output_dir=output_dir,
        is_label=True
    )
    
    # Save results to CSV
    image_df = multi_input_adapter.results_to_dataframe(image_results)
    label_df = multi_input_adapter.results_to_dataframe(label_results)
    
    # Save to CSV files
    csv_output_dir = os.path.join(output_dir, 'metadata')
    Path(csv_output_dir).mkdir(parents=True, exist_ok=True)
    
    image_df.to_csv(os.path.join(csv_output_dir, 'image_patches.csv'), index=False)
    label_df.to_csv(os.path.join(csv_output_dir, 'label_patches.csv'), index=False)
    
    print("Processing complete!")
    print(f"Images processed: {len(image_results['patches'])}")
    print(f"Labels processed: {len(label_results['patches'])}")
    print(f"Metadata saved to: {csv_output_dir}")