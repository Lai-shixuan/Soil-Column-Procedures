import sys
# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
from API_functions.Soils import threshold_position_independent as tpi
from API_functions.DL import multi_input_adapter
from API_functions import file_batch as fb

import numpy as np
import cv2


def precheck(dataset: list[np.ndarray], labels: list[np.ndarray], target_size: int = 512, stride: int = 512, verbose: bool = True) -> dict:
    """
    Preprocess the dataset and labels by:
    1. Padding smaller images to the target size if needed.
    2. Converting images to 8-bit.
    3. Ensuring labels contain only values 0 and 255.
    4. Handling large images by splitting them using a sliding window technique.
    
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
    padded_count = 0
    converted_count = 0
    thresholded_count = 0
    sliding_window_image_count = 0
    sliding_window_patch_count = 0

    # Processing each image and label in a single loop
    for img, label in zip(dataset, labels):
        h, w = img.shape  # Using h, w for height and width of the image

        # 1. Padding images and labels smaller than target_size
        if h < target_size or w < target_size:
            img = multi_input_adapter.padding_img(input=img, target_size=target_size, color=255)
            label = multi_input_adapter.padding_img(input=label, target_size=target_size, color=0)
            padded_count += 1  # Combined count for padding images and labels

        # 2. Convert to 8-bit if necessary
        if img.dtype != 'uint8' or label.dtype != 'uint8':
            img = fb.bitconverter.convert_to_8bit_one_image(img)
            label = fb.bitconverter.convert_to_8bit_one_image(label)
            converted_count += 1

        # 3. Threshold label to ensure it's binary (0 and 255)
        if set(label.flatten()) != {0, 255}: 
            label = tpi.user_threshold(image=label, optimal_threshold=255//2)
            thresholded_count += 1

        # 4. Sliding window to create patches from large images
        h, w = img.shape
        if h > target_size or w > target_size:
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

    # Print counts if verbose is enabled
    if verbose:
        print(f'Images and labels padded: {padded_count}')
        print(f'Images converted to 8-bit: {converted_count}')
        print(f'Labels thresholded to binary: {thresholded_count}')
        print(f'Images split using sliding window: {sliding_window_image_count}')
        print(f'It created {sliding_window_patch_count} patches')

    # Return the results as a dictionary
    return {
        'patches': patches,
        'patch_labels': patch_labels,
        'original_image_info': original_image_info
    }


def test_precheck():
    test_paths = fb.get_image_names('g:/DL_Data_raw/Unit_test/precheck/test_images/', None, 'png')
    test_labels_paths = fb.get_image_names('g:/DL_Data_raw/Unit_test/precheck/test_labels/', None, 'png')

    tests = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in test_paths]
    test_labels = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in test_labels_paths]
    
    datasets = precheck(tests, test_labels)

    # print every img in datasets['patches']
    for i, img in enumerate(datasets['patches']):
        cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patches/{i}.png', img)
    for i, img in enumerate(datasets['patch_labels']):
        cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patch_labels/{i}.png', img)


if __name__ == '__main__':
    test_precheck()