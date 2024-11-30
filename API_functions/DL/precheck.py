import sys
# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
from API_functions.Soils import threshold_position_independent as tpi
from API_functions.DL import multi_input_adapter
from API_functions import file_batch as fb

import numpy as np


def _harmonized_bit_number(img: np.ndarray) -> np.ndarray:
    """
    Convert the images to float32 format.
    For images with multiple channels, the function will issue a warning and convert to grayscale.
    For images with uint16 datatype, the function will convert them to 8-bit images float.
    For images with uint8 datatype, the function will change the datatype to float32.
    """
    
    match (len(img.shape), img.dtype):
        case (2, 'uint16'):
            return fb.bitconverter.convert_to_8bit_one_image(img, type='float')
        case (2, 'uint8'):
            img = img.astype('float32')
            return img
        case (2, 'float32'):
            return img
        case (dims, _) if dims != 2:
            raise ValueError('The images should be grayscale')
        case _:
            raise ValueError('I do not know how to handle this image')


def _append_patches(img: np.ndarray, target_size: int, stride: int) -> tuple[list, tuple, dict]:
    """Helper function to handle patch creation and appending"""
    h, w = img.shape
    patches = []
    stats = {'is_split': False, 'is_padded': False, 'patch_count': 0}
    
    if h < target_size or w < target_size:
        # Pad small images
        patches.append(multi_input_adapter.padding_img(img, target_size, color=255))
        stats['is_padded'] = True
        stats['patch_count'] = 1
    elif h > target_size or w > target_size:
        # Split large images
        patches.extend(multi_input_adapter.sliding_window(img, target_size, stride))
        stats['is_split'] = True
        stats['patch_count'] = len(patches)
    else:
        # Image is exactly target size
        patches.append(img)
        stats['patch_count'] = 1
        
    return patches, (h, w), stats

def precheck(dataset: list[np.ndarray], labels: list[np.ndarray], target_size: int = 512, stride: int = 512, verbose: bool = True) -> dict:
    """
    Preprocess the dataset and labels by:
    1. Converting images to float32 format, and the maximum value to 255.
    2. Ensuring labels contain only values 0 and 255.
    3.1 Padding smaller images to the target size if needed.
    3.2 Handling large images by splitting them using a sliding window technique.
    4. Converting images to 0-1 range.
    
    Parameters:
    - dataset (list[np.ndarray]): List of input images (grayscale).
    - labels (list[np.ndarray]): List of label images (grayscale).
    - target_size (int): Target patch size for splitting large images.
    - stride (int): Stride for sliding window (equal to target_size).
    - verbose (bool): If True, print detailed processing logs.
    
    Returns:
    - dict: Dictionary containing:
        - 'patches': List of image patches.
        - 'patch_labels': List of label patches.
        - 'original_image_info': List of original image sizes.
    """
    if len(dataset) != len(labels):
        raise ValueError('The number of images in the dataset does not match the number of images in the labels')
    
    patches, patch_labels, original_image_info = [], [], []
    stats = {'thresholded': 0, 'padded': 0, 'split': 0, 'total_patches': 0}

    # Processing each image and label in a single loop
    for img, label in zip(dataset, labels):
        # 1. Convert images to float32
        img = _harmonized_bit_number(img)
        label = _harmonized_bit_number(label)

        # 2. Threshold label if needed
        if set(label.flatten()) != {0, 255}:
            label = tpi.user_threshold(image=label, optimal_threshold=255//2)
            stats['thresholded'] += 1

        # 3. Handle patches
        new_patches, img_info, patch_stats = _append_patches(img, target_size, stride)
        new_labels, _, _ = _append_patches(label, target_size, stride)
        
        # 4. Convert to 0-1 range
        new_patches = [fb.bitconverter.grayscale_to_binary_one_image(p) for p in new_patches]
        new_labels = [fb.bitconverter.grayscale_to_binary_one_image(l) for l in new_labels]
        
        patches.extend(new_patches)
        patch_labels.extend(new_labels)
        original_image_info.extend([img_info] * len(new_patches))
        
        # Update statistics
        if patch_stats['is_split']:
            stats['split'] += 1
        if patch_stats['is_padded']:
            stats['padded'] += 1
        if patch_stats['patch_count'] > 0:
            stats['total_patches'] += patch_stats['patch_count']

    if verbose:
        print(f'Labels thresholded to binary: {stats["thresholded"]}')
        print(f'Images and labels padded: {stats["padded"]}')
        print(f'Images split using sliding window: {stats["split"]}')
        print(f'Total patches created: {stats["total_patches"]}')

    return {
        'patches': patches,
        'patch_labels': patch_labels,
        'original_image_info': original_image_info
    }
