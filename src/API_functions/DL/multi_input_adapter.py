import numpy as np
import sys
import warnings
import cv2
import pandas as pd
from enum import Enum

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from src.API_functions.Soils import threshold_position_independent as tpi
from src.API_functions.Images import file_batch as fb
from src.API_functions.DL import shape_processor as processor
from src.API_functions.DL import shape_detectors as detector
from tqdm import tqdm


def padding_img(input: np.ndarray, target_size: int, color: int) -> tuple[np.ndarray, dict]:
    """Pad a grayscale image to target size with padding information.
    
    Algorithm:
    1. Calculate padding sizes for all sides to center the image
    2. Apply padding with constant values
    3. Return both padded image and padding information
    
    Args:
        input: Input grayscale image
        target_size: Desired output size (square)
        color: Padding color value (0 for black, 1 for white)
    
    Returns:
        tuple: (padded_image, padding_info_dict)
        - padded_image: Image padded to target_size
        - padding_info_dict: Dictionary containing padding sizes and color
    
    Raises:
        ValueError: If input image is larger than target size
    """
    if input.shape[0] > target_size or input.shape[1] > target_size:
        raise ValueError("Input image size is larger than target size")

    padding_top = (target_size - input.shape[0]) // 2
    padding_left = (target_size - input.shape[1]) // 2
    padding_bottom = target_size - input.shape[0] - padding_top
    padding_right = target_size - input.shape[1] - padding_left

    output = np.pad(input, ((padding_top, padding_bottom),
                           (padding_left, padding_right)),
                   mode='constant', constant_values=color)
    
    padding_info = {
        'top': padding_top,
        'bottom': padding_bottom,
        'left': padding_left,
        'right': padding_right,
        'color': color
    }
    
    return output, padding_info


def restore_image_batch(datasets: dict, target_size: int = 512):
    """Restore complete images from their patches and metadata.
    
    Algorithm:
    1. Group patches by original image
    2. Process each image:
        a. Single patch case:
           - Remove padding if present
           - Restore to original dimensions
        b. Multiple patches case:
           - Reconstruct using positions
           - Handle overlaps by averaging
    
    Args:
        datasets: Dictionary containing:
            - patches: List of image patches
            - patch_positions: Patch position coordinates
            - patch_to_image_map: Mapping to original images
            - original_image_info: Original image dimensions
            - padding_info: Optional padding information
        target_size: Size of input patches
    
    Returns:
        list: Restored images in original dimensions
    
    Raises:
        ValueError: If required keys missing from datasets
    """
    images_restrored = []
    
    # Validate input data
    if not all(key in datasets for key in ['patches', 'patch_positions', 'patch_to_image_map', 'original_image_info']):
        raise ValueError("Missing required keys in datasets dictionary")

    # Get number of patches for each image
    patch_counts = {}
    for idx in datasets['patch_to_image_map']:
        patch_counts[idx] = patch_counts.get(idx, 0) + 1

    # Process each original image
    for img_idx, count in patch_counts.items():
        if count == 1:
            # Single patch case
            patch_idx = datasets['patch_to_image_map'].index(img_idx)
            patch = datasets['patches'][patch_idx]
            # Handle padding if present
            if 'padding_info' in datasets and datasets['padding_info'][patch_idx]:
                padding_info = datasets['padding_info'][patch_idx]
                h, w = datasets['original_image_info'][img_idx]
                patch = patch[padding_info['top']:target_size-padding_info['bottom'],
                            padding_info['left']:target_size-padding_info['right']]
            images_restrored.append(patch)
        else:
            # Multiple patches case
            image_patches = [
                patch for patch, map_idx in zip(datasets['patches'], datasets['patch_to_image_map'])
                if map_idx == img_idx
            ]
            # Get positions for this image's patches
            image_positions = []
            idx_range = range(len(datasets['patches']))
            for idx in idx_range:
                if datasets['patch_to_image_map'][idx] == img_idx:
                    pos = datasets['patch_positions'][idx]
                    image_positions.append(pos)
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
    - patch_positions: List of (row, col) positions for each patch
    - image_shape: Original image shape (height, width)
    - target_size: Size of each patch
    """
    h, w = image_shape
    output = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for patch, (row, col) in zip(patches, patch_positions):
        # Get actual patch dimensions
        ph, pw = patch.shape
        
        # Calculate valid region sizes (handle boundary conditions)
        valid_h = min(ph, h - row)
        valid_w = min(pw, w - col)
        
        # Add patch to output using only valid regions
        output[row:row + valid_h, col:col + valid_w] += patch[:valid_h, :valid_w]
        count[row:row + valid_h, col:col + valid_w] += 1
    
    # Avoid division by zero
    mask = count > 0
    output[mask] /= count[mask]
    
    return output


def sliding_window(input: np.ndarray, target_size: int, stride: int, return_positions: bool = False) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split image into patches using sliding window approach.
    
    Algorithm:
    1. Main grid processing:
       - Move window by stride steps across image
       - Extract patches and record positions
    2. Edge handling:
       a. Process rightmost column if partial
       b. Process bottom row if partial
       c. Handle bottom-right corner if needed
    
    Args:
        input: Input image to be split
        target_size: Size of each patch
        stride: Step size between patches
        return_positions: Whether to return patch positions
    
    Returns:
        If return_positions:
            tuple: (patches, positions)
            - patches: List of image patches
            - positions: List of (row, col) coordinates
        Else:
            list: Image patches only
    """
    patches = []
    positions = []  # Store (row, col) positions of each patch
    h, w = input.shape[:2]
    
    for row in range(0, h - target_size + 1, stride):
        for col in range(0, w - target_size + 1, stride):
            patch = input[row:row + target_size, col:col + target_size]
            patches.append(patch)
            positions.append((row, col))
    
    # Handle the rightmost part
    if w % target_size != 0:
        for row in range(0, h - target_size + 1, stride):
            patch = input[row:row + target_size, -target_size:]
            patches.append(patch)
            positions.append((row, w - target_size))
    
    # Handle the bottom part
    if h % target_size != 0:
        for col in range(0, w - target_size + 1, stride):
            patch = input[-target_size:, col:col + target_size]
            patches.append(patch)
            positions.append((h - target_size, col))
    
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


def _append_patches(img: np.ndarray, target_size: int, stride: int, is_label: bool) -> tuple[list, tuple, dict, list, list]:
    """Process and split/pad image into patches based on size requirements.
    
    Algorithm:
    1. Check image dimensions against target size
    2. Case handling:
        a. If image smaller: pad to target size
        b. If image larger: split using sliding window
        c. If image exact size: use as-is
    3. Track padding information and statistics
    
    Args:
        img: Input grayscale image
        target_size: Desired patch size
        stride: Step size for sliding window
        is_label: Whether image is a label (affects padding color)
    
    Returns:
        tuple: (patches, image_info, stats, positions, padding_info_list)
        - patches: List of image patches
        - image_info: Original image dimensions (h, w)
        - stats: Dictionary of processing statistics
        - positions: List of (row, col) positions for each patch
        - padding_info_list: List of padding information for each patch
    """
    h, w = img.shape
    patches = []
    patch_positions = []
    padding_info_list = []
    stats = {'is_split': False, 'is_padded': False, 'patch_count': 0}
    padding_color = 0 if is_label else 1
    
    if h < target_size or w < target_size:
        # Case 1: Image smaller than target - pad it
        patch, padding_info = padding_img(img, target_size, color=padding_color)
        patches.append(patch)
        patch_positions.append((0, 0))
        padding_info_list.append(padding_info)
        stats['is_padded'] = True
        stats['patch_count'] = 1
        
    elif h > target_size or w > target_size:
        # Case 2: Image larger than target - split it
        split_patches, split_positions = sliding_window(img, target_size, stride, return_positions=True)
        patches.extend(split_patches)
        patch_positions.extend(split_positions)
        padding_info_list.extend([None] * len(split_patches))
        stats['is_split'] = True
        stats['patch_count'] = len(split_patches)
        
    else:
        # Case 3: Image exactly target size - use as is
        patches.append(img)
        patch_positions.append((0, 0))
        padding_info_list.append(None)
        stats['patch_count'] = 1
        
    return patches, (h, w), stats, patch_positions, padding_info_list


class DetectionMode(Enum):
    NONE = "none"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"

def precheck(
    images: list[np.ndarray], 
    detection_mode: DetectionMode,
    is_label: bool = False, 
    target_size: int = 512, 
    stride: int = 512, 
    verbose: bool = True, 
) -> dict:
    """Preprocess images for deep learning pipeline.
    
    Algorithm:
    1. For each image:
        a. Apply thresholding if it's a label
        b. Shape detection and ROI extraction based on mode:
           - None: Skip detection
           - Rectangle/Ellipse: Detect and cut to shape
        c. Process into patches:
           - Pad if smaller than target
           - Split if larger than target
           - Use as-is if exact size
    2. Track all metadata and statistics
    
    Args:
        images: List of input images
        detection_mode: Shape detection method (NONE/RECTANGLE/ELLIPSE)
        is_label: Whether images are labels
        target_size: Target patch size
        stride: Step size for sliding window
        verbose: Whether to print statistics
    
    Returns:
        dict: Processing results containing:
            - patches: Processed image patches
            - patch_positions: Position information
            - original_image_info: Original dimensions
            - patch_to_image_map: Mapping to source images
            - shape_params: Shape detection results
            - padding_info: Padding information
    """
    patches = []  # List of lists for patches
    patch_positions = []  # Flat list of positions
    original_image_info = []
    shape_params = []  # New list to store shape parameters
    stats = {'thresholded': 0, 'padded': 0, 'split': 0, 'total_patches': 0}

    patch_to_image_map = []  # Track which original image each patch belongs to
    current_image_idx = 0
    padding_info_list = []  # New list to store padding info

    for img in tqdm(images):
        # Apply thresholding only if this is a label image
        if is_label and set(img.flatten()) != {0, 1}:
            img = tpi.user_threshold(image=img, optimal_threshold=1/2)
            stats['thresholded'] += 1

        # Shape detection based on mode
        if detection_mode == DetectionMode.NONE:
            params = None
        else:
            shape_detector = (detector.RectangleDetector() if detection_mode == DetectionMode.RECTANGLE 
                            else detector.EllipseDetector())
            result, params = processor.process_shape_detection(img, shape_detector, is_label=is_label, draw_mask=False)
            img = result['cut']
            img = processor.adjust_image_to_shape(img, params, target_size)
        
        shape_params.append(params)  # Store the shape parameters

        # Handle patches
        new_patches, img_info, patch_stats, positions, padding_infos = _append_patches(img, target_size, stride, is_label)
        
        # Store patches and positions for this image
        patches.extend(new_patches)
        patch_positions.extend(positions)  # Store positions directly
        original_image_info.append(img_info)
        padding_info_list.extend(padding_infos)

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
        'patch_positions': patch_positions,  # Now a flat list of positions
        'original_image_info': original_image_info,
        'patch_to_image_map': patch_to_image_map,
        'shape_params': shape_params,
        'padding_info': padding_info_list  # Add padding info to results
    }


def format_ellipse_params(params):
    """Convert EllipseParams to dictionary"""
    if params is None:
        return None
    return {
        'type': 'ellipse',
        'center_x': params.center[0],
        'center_y': params.center[1],
        'covered_pixels': params.covered_pixels,
        'long_axis': params.long_axis,
        'short_axis': params.short_axis
    }

def format_rectangle_params(params):
    """Convert RectangleParams to dictionary"""
    if params is None:
        return None
    return {
        'type': 'rectangle',
        'center_x': params.center[0],
        'center_y': params.center[1],
        'covered_pixels': params.covered_pixels,
        'width': params.width,
        'height': params.height
    }

def _create_base_row(i: int, img_idx: int, position: tuple, orig_dims: tuple, padding_info: dict = None) -> dict:
    """Create base row data common to all detection modes"""
    row = {
        'patch_index': i,
        'original_image_index': img_idx,
        'position_x': position[0],
        'position_y': position[1],
        'original_width': orig_dims[0],
        'original_height': orig_dims[1]
    }
    
    # Add padding information if available
    if padding_info:
        row.update({
            'padding_top': padding_info['top'],
            'padding_bottom': padding_info['bottom'],
            'padding_left': padding_info['left'],
            'padding_right': padding_info['right'],
            'padding_color': padding_info['color']
        })
    else:
        row.update({
            'padding_top': None,
            'padding_bottom': None,
            'padding_left': None,
            'padding_right': None,
            'padding_color': None
        })
    
    return row

def _add_shape_params(row: dict, params_dict: dict) -> dict:
    """Add shape-specific parameters to row"""
    if params_dict is None:
        row.update({
            'shape_type': 'none',
            'center_x': None,
            'center_y': None,
            'covered_pixels': None
        })
        return row

    row.update({
        'shape_type': params_dict['type'],
        'center_x': params_dict['center_x'],
        'center_y': params_dict['center_y'],
        'covered_pixels': params_dict['covered_pixels']
    })

    # Add shape-specific measurements
    if params_dict['type'] == 'ellipse':
        row.update({
            'long_axis': params_dict['long_axis'],
            'short_axis': params_dict['short_axis'],
            'width': None,
            'height': None
        })
    else:  # rectangle
        row.update({
            'long_axis': None,
            'short_axis': None,
            'width': params_dict['width'],
            'height': params_dict['height']
        })
    return row

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert preprocessing results to structured DataFrame.
    
    Algorithm:
    1. For each patch:
        a. Create base row with patch/image information
        b. Add padding information if present
        c. Add shape parameters based on detection mode
        d. Convert all parameters to appropriate format
    2. Combine into structured DataFrame
    
    Args:
        results: Dictionary containing preprocessing results
    
    Returns:
        pd.DataFrame: Structured data with columns:
            - Patch information (index, position)
            - Original image information
            - Shape parameters (if any)
            - Padding information (if any)
    """
    data = []
    n_patches = len(results['patches'])
    patch_counts = {}
    
    for i in range(n_patches):
        img_idx = results['patch_to_image_map'][i]
        patch_counts[img_idx] = patch_counts.get(img_idx, 0)
        position = results['patch_positions'][i]  # Access position directly from flat list
        patch_counts[img_idx] += 1
        
        orig_dims = results['original_image_info'][img_idx]
        
        # Create base row
        row = _create_base_row(i, img_idx, position, orig_dims, results['padding_info'][i])
        
        # Handle shape parameters based on detection mode
        shape_params = results['shape_params'][img_idx]
        if shape_params is None:
            params_dict = None
        else:
            params_dict = (format_ellipse_params(shape_params) 
                        if hasattr(shape_params, 'long_axis')
                        else format_rectangle_params(shape_params))
        
        # Add shape parameters to row
        row = _add_shape_params(row, params_dict)
        data.append(row)
    
    return pd.DataFrame(data)