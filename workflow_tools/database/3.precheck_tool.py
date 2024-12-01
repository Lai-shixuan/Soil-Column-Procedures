import cv2
import logging
from pathlib import Path
from typing import List, Union
import os

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
    patch_dir = os.path.join(output_dir, 'patch_labels' if is_label else 'patches')
    Path(patch_dir).mkdir(parents=True, exist_ok=True)

    # Read images
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)
            else:
                logger.warning(f"Could not read image: {path}")
        except Exception as e:
            logger.error(f"Error reading {path}: {str(e)}")
            continue

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
    # Example usage
    import glob
    
    # Example paths - modify these according to your data location
    image_dir = "path/to/your/images"
    label_dir = "path/to/your/labels"
    output_dir = "path/to/output"
    
    # Get all image files
    image_paths = glob.glob(f"{image_dir}/*.png")  # adjust file extension as needed
    label_paths = glob.glob(f"{label_dir}/*.png")
    
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