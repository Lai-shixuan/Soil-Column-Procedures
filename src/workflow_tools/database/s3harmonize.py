import cv2
import logging
import os
import sys

from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from src.API_functions.Images import file_batch as fb
from src.API_functions.DL.multi_input_adapter import harmonized_normalize

def batch_harmonize_and_save(image_paths: list, output_dir: str):
    """
    Harmonize a batch of images and save them to the output directory
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save the harmonized images
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths):
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                continue

            # Harmonize image
            harmonized = harmonized_normalize(img)

            # Create output path, change the suffix
            filename = os.path.basename(image_path)
            filename = filename.replace("cutted", "harmonized")

            # Change to tif
            output_path = os.path.join(output_dir, f"{filename}")

            # Save harmonized image
            cv2.imwrite(output_path, harmonized)

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # for i in range(22, 28):
        # Example paths - modify these according to your data location
        # image_dir = f'f:/3.Experimental_Data/Soils/Dongying_normal/Soil.column.{i:04d}/2.ROI/'
        # output_dir = f'f:/3.Experimental_Data/Soils/Dongying_normal/Soil.column.{i:04d}/3.Harmonized/image'
    image_dir = r'f:\3.Experimental_Data\Soils\Online\Soil.column.0035\2.ROI\image'
    output_dir = r'f:\3.Experimental_Data\Soils\Online\Soil.column.0035\3.Harmonized\image'
    
    # Get all image files
    image_paths = fb.get_image_names(image_dir, None, 'tif')
    
    # print(f"Processing column {i}...")

    # Change the 'png' to 'tif' in the output directory
    # Change the 'cutted' to 'harmonized' in the output directory
    batch_harmonize_and_save(image_paths, output_dir)

    # print(f"Completed column {i}")
