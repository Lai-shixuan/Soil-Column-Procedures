import sys
import warnings
# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
from API_functions.Soils import threshold_position_independent as tpi
from API_functions.DL import multi_input_adapter
from API_functions import file_batch as fb

import numpy as np


def harmonized_bit_number(img: np.ndarray) -> np.ndarray:
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
            warnings.warn(f'RGB image detected', UserWarning)
        case _:
            raise ValueError('The images should be grayscale')


def precheck(dataset: list[np.ndarray], labels: list[np.ndarray], target_size: int = 512, stride: int = 512, verbose: bool = True) -> dict:
    """
    Preprocess the dataset and labels by:
    1. Converting images to float32 format, and the maximum value to 255.
    2. Converting images to 0-1 range.
    3. Padding smaller images to the target size if needed.
    4. Ensuring labels contain only values 0 and 255.
    5. Handling large images by splitting them using a sliding window technique.
    
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
    
    patches = []
    patch_labels = []
    original_image_info = []

    # Initialize counters for different operations
    thresholded_count = 0
    padded_count = 0
    sliding_window_image_count = 0
    sliding_window_patch_count = 0

    # Processing each image and label in a single loop
    for img, label in zip(dataset, labels):
        h, w = img.shape  # Using h, w for height and width of the image

        # 1. Check what type of images are provided, and convert them to float32
        img = harmonized_bit_number(img)
        label = harmonized_bit_number(label)

        # 2. Threshold label to ensure it's binary (0 and 255)
        if set(label.flatten()) != {0, 255}: 
            label = tpi.user_threshold(image=label, optimal_threshold=255//2)
            thresholded_count += 1

        # 3.1 Padding images and labels smaller than target_size
        # 3.2 Splitting images and labels larger than target_size
        if h < target_size or w < target_size:
            img = multi_input_adapter.padding_img(input=img, target_size=target_size, color=255)
            label = multi_input_adapter.padding_img(input=label, target_size=target_size, color=0)
            padded_count += 1  # Combined count for padding images and labels
        elif h > target_size or w > target_size:
            img_patches = multi_input_adapter.sliding_window(img, target_size, stride)
            label_patches = multi_input_adapter.sliding_window(label, target_size, stride)
            patches.extend(img_patches)
            patch_labels.extend(label_patches)
            original_image_info.extend([(h, w)] * len(img_patches))
            sliding_window_image_count += 1
            sliding_window_patch_count += len(img_patches)
        else:
            patches.append(img)
            patch_labels.append(label)
            original_image_info.append((h, w))

        # 4. Convert to 0-1 images
        img = fb.bitconverter.grayscale_to_binary_one_image(img)
        label = fb.bitconverter.grayscale_to_binary_one_image(label)

    # Print counts if verbose is enabled
    if verbose:
        print(f'Labels thresholded to binary: {thresholded_count}')
        print(f'Images and labels padded: {padded_count}')
        print(f'Images split using sliding window: {sliding_window_image_count}')
        print(f'It created {sliding_window_patch_count} patches')

    # Return the results as a dictionary
    return {
        'patches': patches,
        'patch_labels': patch_labels,
        'original_image_info': original_image_info
    }
