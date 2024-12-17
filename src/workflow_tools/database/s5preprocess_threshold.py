import numpy as np
import cv2 as cv
import sys

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from tqdm import tqdm
from pathlib import Path
from src.API_functions.Images import file_batch as fb
from src.API_functions.Soils import pre_process
from src.API_functions.Soils import threshold_position_independent as tmi


def batch_process_images(path_in, path_out, process_function, file_pattern: str=None, extension: str='tif', pattern_include: bool=False):
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

        output_path = path_out / Path(image_path).name.replace('harmonized', 'preprocessed')
        cv.imwrite(str(output_path), processed_image)


if __name__ == "__main__":

    def process_pipeline(image):
        # image = pre_process.median(image, 5)
        # image = pre_process.clahe(image)
        # image = 1 - np.mean(image)

        # image = pre_process.median(image, 5)
        # image2 = image.copy()
        # _, threshold = tmi.kmeans_3d(image2, return_threshold=True)
        # image = tmi.user_threshold(image, threshold-0.15)

        image = 1 - image
        return image

    path_in = Path(r'g:\DL_Data_raw\version6-large\4.Converted\label')
    path_out = Path(r'g:\DL_Data_raw\version6-large\4.Converted\label')
    extension = 'tif'
    
    batch_process_images(
        path_in=path_in,
        path_out=path_out,
        process_function=process_pipeline,

        # Select only '0035' or exclude '0035'
        file_pattern=None,
        pattern_include=True,

        extension=extension
    )
