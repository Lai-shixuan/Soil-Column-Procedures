import cv2
import numpy as np

def resize_image(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize image to target dimensions while maintaining aspect ratio.
    
    The image is resized to fit within the target dimensions while maintaining
    its aspect ratio. The result is centered on a black canvas of the target size.
    
    Args:
        img: Input image as numpy array
        target_width: Desired width of output image
        target_height: Desired height of output image
    
    Returns:
        Resized image centered on black canvas of target size
    """
    h, w = img.shape[:2]
    # Calculate scaling factor to fit within target size
    scale = min(target_height/h, target_width/w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create black canvas of target size
    if len(img.shape) == 3:
        canvas = np.zeros((target_height, target_width, 3), dtype=img.dtype)
    else:
        canvas = np.zeros((target_height, target_width), dtype=img.dtype)
        
    # Calculate position to paste resized image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Fix the incorrect offset addition
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas