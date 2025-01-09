import numpy as np
import cv2 as cv
import sys
import csv

# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from tqdm import tqdm
from pathlib import Path
from src.API_functions.Images import file_batch as fb
from src.API_functions.Soils import pre_process
from src.API_functions.Soils import threshold_position_independent as tmi


def batch_process_images(path_in, path_out, process_function,
                        original_suffix: str, new_suffix: str, file_pattern: str=None,
                        extension: str='tif', pattern_include: bool=False):
    """
    Batch process images with a sequence of processing steps
    """
    path_in = Path(path_in)
    path_out = Path(path_out)
    image_lists = fb.get_image_names(str(path_in), None, extension)
    
    if file_pattern:
        if pattern_include:
            image_lists = [item for item in image_lists if Path(item).name.startswith(file_pattern)]
        else:
            image_lists = [item for item in image_lists if not Path(item).name.startswith(file_pattern)]

    path_out.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_lists):
        image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
        processed_image = process_function(image)

        output_path = path_out / Path(image_path).name.replace(original_suffix, new_suffix)
        cv.imwrite(str(output_path), processed_image)

    offset = False

    # Todo: using offset in main function configuration
    if offset:
        csv_path = path_out / "threshold_offsets.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "offset"])
            for image_path in tqdm(image_lists):
                image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
                processed_image, offset = process_function(image)

                output_path = path_out / Path(image_path).name.replace(original_suffix, new_suffix)
                cv.imwrite(str(output_path), processed_image)
                writer.writerow([Path(image_path).name, offset])


def evaluate_threshold_offset(image, threshold):
    step = 0.001
    offset = 0.12
    max_steps = 200
    for _ in range(max_steps):
        bin_img = tmi.user_threshold(image, threshold - offset)
        bin_img = 1 - bin_img
        white_fraction = bin_img.mean()
        if 0.03 <= white_fraction <= 0.35:
            return round(offset, 3)
        if white_fraction < 0.03:
            offset = max(0, offset - step)
        else:
            offset = min(0.2, offset + step)
    print("Warning: Could not find suitable offset in [0, 0.2].")
    return round(offset, 3)


if __name__ == "__main__":

    def process_pipeline(image):
        """User defined processing pipeline"""
        # image = pre_process.reduce_gaussian_noise(image, strength=1, use_gaussian=True)

        # image = pre_process.bm3d_denoising(image, sigma_psd=0.5)

        image = pre_process.median(image, 5)
        image = pre_process.reduce_poisson_noise(image, strength=3)
        # image = pre_process.clahe_float32(image)
        image = image - np.mean(image)

        # image = pre_process.median(image, 5)
        # image2 = image.copy()
        # _, threshold = tmi.kmeans_3d(image2, return_threshold=True)
        # offset = evaluate_threshold_offset(image2, threshold)
        # image = tmi.user_threshold(image, threshold - offset)
        # image = 1 - image

        return image 

    path_in = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/image')
    path_out = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/5.2.Preprocessed')
    read_img_extension = 'tif'
    
    batch_process_images(
        path_in=path_in,
        path_out=path_out,
        process_function=process_pipeline,

        # Select only '0035' or exclude '0035'
        file_pattern=None,
        pattern_include=True,

        # Name replaces
        extension=read_img_extension,
        original_suffix='harmonized',
        new_suffix='preprocessed'
    )
