import numpy as np
from typing import Tuple
import sys

sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

def pad_data_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Pad data image to target size with mean value."""
    h, w = image.shape
    padded = np.zeros(target_size, dtype=image.dtype)
    
    # Calculate padding
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    
    # Copy image to padded array
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    # Fill padding with mean
    image_mean = np.mean(image)
    mask = np.zeros(target_size, dtype=np.float32)
    mask[pad_h:pad_h+h, pad_w:pad_w+w] = 1.0
    padded[mask == 0] = image_mean
    
    return padded

def pad_label_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Pad label image to target size with zeros."""
    h, w = image.shape
    padded = np.zeros(target_size, dtype=image.dtype)
    
    # Calculate padding
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    
    # Copy image to padded array
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    return padded

def get_padding_mask(image_shape: Tuple[int, int], target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Create mask indicating original image area."""
    h, w = image_shape
    mask = np.zeros(target_size, dtype=np.float32)
    
    # Calculate padding
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    
    # Mark original image area
    mask[pad_h:pad_h+h, pad_w:pad_w+w] = 1.0
    
    return mask
