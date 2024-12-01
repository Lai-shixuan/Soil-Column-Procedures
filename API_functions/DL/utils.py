import numpy as np


def check_corners(image: np.ndarray, corner_size: int = 4) -> bool:
    """Check if all four corners contain white pixels"""
    h, w = image.shape
    corners = [
        image[0:corner_size, 0:corner_size],
        image[0:corner_size, w-corner_size:w],
        image[h-corner_size:h, 0:corner_size],
        image[h-corner_size:h, w-corner_size:w]
    ]
    return all(np.any(corner == 1) for corner in corners)


def create_circular_mask(image: np.ndarray) -> np.ndarray:
    """Create a circular mask that fits within the image"""
    h, w = image.shape
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = np.zeros_like(image)
    mask[dist_from_center <= radius] = 1
    return mask