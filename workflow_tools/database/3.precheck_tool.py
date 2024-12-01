import cv2
import logging
import os
import glob
import sys

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from typing import List, Union
from API_functions import file_batch as fb
from API_functions.DL import multi_input_adapter

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

    # Save patches
    for i, img in enumerate(results['patches']):
        output_path = os.path.join(patch_dir, f'{i}.tif')
        cv2.imwrite(output_path, img)

    # Log information
    logger.info("Precheck Results:")
    logger.info(f"Patch positions: {results['patch_positions']}")
    logger.info(f"Original image info: {results['original_image_info']}")
    logger.info(f"Patch to image map: {results['patch_to_image_map']}")
    logger.info(f"Shape parameters: {results['shape_params']}")

    return results

if __name__ == "__main__":
    
    # Example paths - modify these according to your data location
    image_dir = "f:/3.Experimental_Data/Soils/Online/Soil.column.0035/2.ROI/image/"
    label_dir = "f:/3.Experimental_Data/Soils/Online/Soil.column.0035/2.ROI/label/"
    output_dir = "f:/3.Experimental_Data/Soils/Online/Soil.column.0035/3.Precheck/"
    
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
    
    print("Processing complete!")
    print(f"Images processed: {len(image_results['patches'])}")
    print(f"Labels processed: {len(label_results['patches'])}")