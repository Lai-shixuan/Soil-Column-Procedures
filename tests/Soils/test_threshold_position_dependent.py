import numpy as np
import cv2
import pytest
import os
from src.API_functions.Soils.threshold_position_dependent import watershed, local_threshold
from src.API_functions.Images import file_batch as fb

@pytest.fixture
def input_folder():
    return 'g:/DL_Data_raw/Unit_test/threshold/images/'

@pytest.fixture
def output_folder():
    folder = 'g:/DL_Data_raw/Unit_test/threshold/threshold/'
    os.makedirs(folder, exist_ok=True)
    return folder

@pytest.fixture
def test_images(input_folder):
    img_paths = fb.get_image_names(input_folder, None, 'tif')
    imgs = fb.read_images(img_paths, gray='gray', read_all=True)
    return imgs, img_paths

def save_result(result, filename, output_folder):
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result)

def validate_watershed(result, original_shape):
    assert isinstance(result, np.ndarray)
    assert result.shape == original_shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})

def validate_local_threshold(result, original_shape, original_dtype):
    """Validate local threshold output"""
    assert isinstance(result, np.ndarray)
    assert result.shape == original_shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})  # uint8 二值图像应该是0和255

def test_watershed(test_images, output_folder):
    imgs, img_paths = test_images
    for i, img in enumerate(imgs):
        result = watershed(img)
        validate_watershed(result, img.shape)
        
        filename = f'watershed_{os.path.basename(img_paths[i])}'
        save_result(result, filename, output_folder)

@pytest.mark.parametrize("block_size,C", [
    (11, 2),
    (21, 2),
    (11, 4)
])

def test_local_threshold(test_images, output_folder, block_size, C):
    imgs, img_paths = test_images
    for i, img in enumerate(imgs):
        img = fb.bitconverter.binary_to_grayscale_one_image(img, 'uint8') 
        result = local_threshold(img, block_size=block_size, C=C)
        validate_local_threshold(result, img.shape, img.dtype)
        
        filename = f'local_thresh_b{block_size}_C{C}_{os.path.basename(img_paths[i])}'
        save_result(result, filename, output_folder)
