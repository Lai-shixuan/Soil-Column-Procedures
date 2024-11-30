import numpy as np
import sys

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from API_functions.Soils import threshold_position_independent as tpi
from API_functions import file_batch as fb


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
            images_restrored.append(datasets['image_patches'][patch_idx])
        else:
            # Image was split into multiple patches, needs restoration
            image_patches = [
                patch for patch, map_idx in zip(datasets['image_patches'], datasets['patch_to_image_map'])
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


def _append_patches(img: np.ndarray, target_size: int, stride: int) -> tuple[list, tuple, dict, list]:
    """Helper function to handle patch creation and appending"""
    h, w = img.shape
    patches = []
    patch_positions = []  # Store (y, x) positions of each patch
    stats = {'is_split': False, 'is_padded': False, 'patch_count': 0}
    
    if h < target_size or w < target_size:
        patches.append(padding_img(img, target_size, color=255))
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
    1. Converting images to float32 format, and the maximum value to 255.
    2. (Only for labels) Ensuring labels contain only values 0 and 255.
    3.1 Padding smaller images to the target size if needed.
    3.2 Handling large images by splitting them using a sliding window technique.
    4. Converting images to 0-1 range.
    
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
    """
    patches = []  # List of lists for patches
    patch_positions = []  # List of lists for patch positions
    original_image_info = []
    stats = {'thresholded': 0, 'padded': 0, 'split': 0, 'total_patches': 0}

    patch_to_image_map = []  # Track which original image each patch belongs to
    current_image_idx = 0

    for img in images:
        # Convert images to float32
        img = _harmonized_bit_number(img)

        # Apply thresholding only if this is a label image
        if is_label and set(img.flatten()) != {0, 255}:
            img = tpi.user_threshold(image=img, optimal_threshold=255//2)
            stats['thresholded'] += 1

        # Handle patches
        new_patches, img_info, patch_stats, positions = _append_patches(img, target_size, stride)
        
        # Convert to 0-1 range
        new_patches = [fb.bitconverter.grayscale_to_binary_one_image(p) for p in new_patches]
        
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
        'patch_to_image_map': patch_to_image_map
    }