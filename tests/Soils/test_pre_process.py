import sys
import pytest
import cv2

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from src.API_functions.Soils import pre_process
from src.API_functions.Images import file_batch as fb

def test_clahe_float32():
    input_folder = Path(r'g:\DL_Data_raw\Unit_test\preprocess\image')
    output_folder = Path(r'g:\DL_Data_raw\Unit_test\preprocess\clahe')

    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    
    img_paths = fb.get_image_names(input_folder, None, 'tif')
    imgs = fb.read_images(img_paths, gray='gray', read_all=True)

    for i, img in enumerate(imgs):
        img_name = Path(img_paths[i]).name
        result = pre_process.clahe_float32(img)
        output_path = output_folder / img_name
        cv2.imwrite(str(output_path), result)
