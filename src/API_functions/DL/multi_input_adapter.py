import numpy as np
import sys
import warnings
import cv2
import pandas as pd

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from API_functions.Soils import threshold_position_independent as tpi
from src.API_functions.Images import file_batch as fb
from API_functions.DL import shape_processor as processor
from API_functions.DL import shape_detectors as detector
from tqdm import tqdm


def padding_img(input: np.ndarray, target_size: int, color: int) -> np.ndarray:
    """
    Only for grayscale image
    """
    if input.shape[0] > target_size or input.shape[1] > target_size:
        raise ValueError("Input image size is larger than target size")

    padding_top = (target_size - input.shape[0]) // 2
    padding_left = (target_size - input.shape[1]) // 2

    output = np.pad(input, ((padding_top, target_size - input.shape[0] - padding_top),
                            (padding_left, target_size - input.shape[1] - padding_left)),
                    mode='constant', constant_values=color)
    
    return output


def restore_image_batch(datasets: dict, target_size: int = 512):
    """Restore the original images from patches and their positions."""

    images_restrored = []

    # Get number of patches for each image
    patch_counts = {}
    for idx in datasets['patch_to_image_map']:
        patch_counts[idx] = patch_counts.get(idx, 0) + 1

    # Process each original image
    for img_idx, count in patch_counts.items():
        if count == 1:
            # Image wasn't split, just save the single patch directly
            patch_idx = datasets['patch_to_image_map'].index(img_idx)
            images_restrored.append(datasets['patches'][patch_idx])
        else:
            # Image was split into multiple patches, needs restoration
            image_patches = [
                patch for patch, map_idx in zip(datasets['patches'], datasets['patch_to_image_map'])
                if map_idx == img_idx
            ]
            image_positions = datasets['patch_positions'][img_idx]
            original_shape = datasets['original_image_info'][img_idx]
            
            restored_image = restore_image(
                patches=image_patches,
                patch_positions=image_positions,
                image_shape=original_shape,
                target_size=target_size
            )
            images_restrored.append(restored_image)
    
    return images_restrored


def restore_image(patches: list[np.ndarray], patch_positions: list[tuple[int, int]], image_shape: tuple[int, int], target_size: int) -> np.ndarray:
    """
    Reconstructs the original image using patches and their positions.
    
    Parameters:
    - patches: List of image patches
    - patch_positions: List of (y, x) positions for each patch
    - image_shape: Original image shape (height, width)
    - target_size: Size of each patch
    """
    h, w = image_shape
    output = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    # Add each patch to the output image 
    for patch, (y, x) in zip(patches, patch_positions):
        output[y:y + target_size, x:x + target_size] += patch
        count[y:y + target_size, x:x + target_size] += 1
    
    # Average out the overlapping regions
    output /= count
    return output


def sliding_window(input: np.ndarray, target_size: int, stride: int, return_positions: bool = False) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """
    # ...existing docstring...
    Returns:
    - If return_positions is False: list[np.ndarray] of patches
    - If return_positions is True: tuple(list[np.ndarray], list[tuple[int, int]]) of patches and their positions
    """
    patches = []
    positions = []  # Store (y, x) positions of each patch
    h, w = input.shape[:2]
    
    for y in range(0, h - target_size + 1, stride):
        for x in range(0, w - target_size + 1, stride):
            patch = input[y:y + target_size, x:x + target_size]
            patches.append(patch)
            positions.append((y, x))
    
    # Handle the rightmost part
    if w % target_size != 0:
        for y in range(0, h - target_size + 1, stride):
            patch = input[y:y + target_size, -target_size:]
            patches.append(patch)
            positions.append((y, w - target_size))
    
    # Handle the bottom part
    if h % target_size != 0:
        for x in range(0, w - target_size + 1, stride):
            patch = input[-target_size:, x:x + target_size]
            patches.append(patch)
            positions.append((h - target_size, x))
    
    # Bottom-right corner
    if h % target_size != 0 and w % target_size != 0:
        patch = input[-target_size:, -target_size:]
        patches.append(patch)
        positions.append((h - target_size, w - target_size))
    
    return (patches, positions) if return_positions else patches


def harmonized_normalize(img: np.ndarray) -> np.ndarray:
    """
    Convert images to float32 format and standardize to grayscale using OpenCV.
    
    Handles:
    - 2D grayscale images
    - 3-channel RGB images
    - 4-channel RGBA images
    - Different bit depths (uint8, uint16)
    
    Returns:
    - np.ndarray: Grayscale float32 image with values between 0 and 1
    """
    
    match img.shape:
        case (_, _):  # height, width
            return fb.bitconverter.grayscale_to_binary_one_image(img)
            
        case (_, _, 3):  # height, width, RGB
            warnings.warn('Converting RGB image to grayscale')
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return fb.bitconverter.grayscale_to_binary_one_image(gray_img)
            
        case (_, _, 4):  # height, width, RGBA
            warnings.warn('Converting RGBA image to grayscale')
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            return fb.bitconverter.grayscale_to_binary_one_image(gray_img)
            
        case _:
            raise ValueError(f'Unsupported image format: shape={img.shape}, dtype={img.dtype}')


def _append_patches(img: np.ndarray, target_size: int, stride: int, is_label: bool) -> tuple[list, tuple, dict, list]:
    """Helper function to handle patch creation and appending"""
    h, w = img.shape
    patches = []
    patch_positions = []  # Store (y, x) positions of each patch
    stats = {'is_split': False, 'is_padded': False, 'patch_count': 0}
    padding_color = 0 if is_label else 1
    
    if h < target_size or w < target_size:
        patches.append(padding_img(img, target_size, color=padding_color))
        patch_positions.append((0, 0))  # Single centered patch
        stats['is_padded'] = True
    elif h > target_size or w > target_size:
        patches, patch_positions = sliding_window(img, target_size, stride, return_positions=True)
        stats['is_split'] = True
        stats['patch_count'] = len(patches)
    else:
        patches.append(img)
        patch_positions.append((0, 0))
        
    return patches, (h, w), stats, patch_positions


def precheck(images: list[np.ndarray], is_label: bool = False, target_size: int = 512, stride: int = 512, verbose: bool = True) -> dict:
    """
    Preprocess the images by:
    1. Converting images to float32 format, and the maximum value to 1.
    2. (Only for labels) Ensuring labels contain only values 0 and 1.
    3. Detecting the shape of the ROI and cutting the image to the shape boundary.
    4.1 Padding smaller images to the target size if needed.
    4.2 Handling large images by splitting them using a sliding window technique.
    
    Parameters:
    - images (list[np.ndarray]): List of input images (grayscale).
    - is_label (bool): If True, treat inputs as label images and apply label-specific processing.
    - target_size (int): Target patch size for splitting large images.
    - stride (int): Stride for sliding window (equal to target_size).
    - verbose (bool): If True, print detailed processing logs.
    
    Returns:
    - dict: Dictionary containing:
        - 'patches': List of all image patches
        - 'patch_positions': List of lists, where each inner list contains (y, x) positions for patches of one image
        - 'original_image_info': List of original image sizes
        - 'patch_to_image_map': List indicating which original image each patch belongs to
        - 'shape_params': List of shape parameters for each image
    """
    patches = []  # List of lists for patches
    patch_positions = []  # List of lists for patch positions
    original_image_info = []
    shape_params = []  # New list to store shape parameters
    stats = {'thresholded': 0, 'padded': 0, 'split': 0, 'total_patches': 0}

    patch_to_image_map = []  # Track which original image each patch belongs to
    current_image_idx = 0

    for img in tqdm(images):
        # Convert images to float32
        img = harmonized_normalize(img)

        # Apply thresholding only if this is a label image
        if is_label and set(img.flatten()) != {0, 1}:
            img = tpi.user_threshold(image=img, optimal_threshold=1/2)
            stats['thresholded'] += 1

        # Auto detect the boundary of the circle or rectangle ROI, if the image is larger than target_size, cut it 
        result, params = processor.process_shape_detection(img, detector.EllipseDetector(), is_label=is_label, draw_mask=False)
        img = result['cut']
        img = processor.adjust_image_to_shape(img, params, target_size)
        shape_params.append(params)  # Store the shape parameters

        # Handle patches
        new_patches, img_info, patch_stats, positions = _append_patches(img, target_size, stride, is_label)
        
        # Store patches and positions for this image
        patches.extend(new_patches)
        patch_positions.append(positions)
        original_image_info.append(img_info)
        
        # Track which patches belong to this image
        patch_to_image_map.extend([current_image_idx] * len(new_patches))
        current_image_idx += 1

        # Update statistics
        if patch_stats['is_split']:
            stats['split'] += 1
        if patch_stats['is_padded']:
            stats['padded'] += 1
        if patch_stats['patch_count'] > 0:
            stats['total_patches'] += patch_stats['patch_count']

    if verbose:
        if is_label:
            print(f'Labels thresholded to binary: {stats["thresholded"]}')
        print(f'Images {"and labels " if is_label else ""}padded: {stats["padded"]}')
        print(f'Images split using sliding window: {stats["split"]} -> {stats["total_patches"]}')

    return {
        'patches': patches,
        'patch_positions': patch_positions,
        'original_image_info': original_image_info,
        'patch_to_image_map': patch_to_image_map,
        'shape_params': shape_params
    }


def format_ellipse_params(params):
    """Convert EllipseParams to dictionary"""
    return {
        'center_x': params.center[0],
        'center_y': params.center[1],
        'covered_pixels': params.covered_pixels,
        'long_axis': params.long_axis,
        'short_axis': params.short_axis
    }

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert precheck results to a DataFrame format with proper handling of nested structures"""
    data = []
    
    # Get total number of patches
    n_patches = len(results['patches'])
    
    for i in range(n_patches):
        # Get image index for this patch
        img_idx = results['patch_to_image_map'][i]
        
        # Get position for this patch from nested structure
        position = results['patch_positions'][img_idx][i % len(results['patch_positions'][img_idx])]
        
        # Get original image dimensions
        orig_dims = results['original_image_info'][img_idx]
        
        # Get shape parameters and convert to dict
        shape_params = format_ellipse_params(results['shape_params'][img_idx])
        
        row = {
            'patch_index': i,
            'original_image_index': img_idx,
            'position_x': position[0],
            'position_y': position[1],
            'original_width': orig_dims[0],
            'original_height': orig_dims[1],
            'center_x': shape_params['center_x'],
            'center_y': shape_params['center_y'],
            'covered_pixels': shape_params['covered_pixels'],
            'long_axis': shape_params['long_axis'],
            'short_axis': shape_params['short_axis']
        }
        data.append(row)
    
    return pd.DataFrame(data)