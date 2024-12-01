import cv2
import numpy as np
import warnings
from typing import Dict, Optional, Tuple, TypeVar

from .shape_base import ShapeDetector, ShapeParams, EllipseParams, RectangleParams
from .log import logger_manager
from .utils import create_circular_mask, check_corners
from API_functions.Soils import threshold_position_independent as tpi
from API_functions import file_batch as fb

T = TypeVar('T', bound=ShapeParams)

class ShapeDrawer:
    """Utility class for drawing shapes on images"""
    @staticmethod
    def draw_shape(image: np.ndarray, params: ShapeParams, 
                    color: Tuple[int, int, int] = (0, 255, 0), 
                    thickness: int = 2) -> np.ndarray:
        result = image.copy()
        if isinstance(params, EllipseParams):
            cv2.ellipse(result, params.center, 
                       (params.long_axis // 2, params.short_axis // 2),
                        0, 0, 360, color, thickness)
        elif isinstance(params, RectangleParams):
            half_width, half_height = params.width // 2, params.height // 2
            top_left = (params.center[0] - half_width, params.center[1] - half_height)
            bottom_right = (params.center[0] + half_width, params.center[1] + half_height)
            cv2.rectangle(result, top_left, bottom_right, color, thickness)
        return result


def cut_with_shape(image: np.ndarray, params: ShapeParams, fillcolor: int) -> np.ndarray:
    """
    Apply detected shape parameters to cut another grayscale image and fill outside pixels with value 1 
    All parametes shrink the shape by 2 pixels to avoid edge artifacts

    Args:
        image (np.ndarray): Input grayscale image to be cut
        params (ShapeParams): Shape parameters (EllipseParams or RectangleParams)

    Returns:
        np.ndarray: Cut grayscale image with outside pixels set to value 1 
        
    Raises:
        ValueError: If input image is not grayscale
    """

    shrinked_params = 2

    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (single channel)")
    
    # Create mask matching the input image dimensions
    mask = np.zeros(image.shape, dtype=np.float32)
    
    # Draw the shape in color (1) on the mask
    if isinstance(params, EllipseParams):
        cv2.ellipse(mask, params.center, 
                    ((params.long_axis - shrinked_params) // 2, (params.short_axis - shrinked_params) // 2),
                    0, 0, 360, 1, -1)
    elif isinstance(params, RectangleParams):
        half_width, half_height = (params.width - shrinked_params) // 2, (params.height - shrinked_params) // 2
        top_left = (params.center[0] - half_width, params.center[1] - half_height)
        bottom_right = (params.center[0] + half_width, params.center[1] + half_height)
        cv2.rectangle(mask, top_left, bottom_right, 1, -1)
    
    # Create output image (all white)
    result = np.full_like(image, fillcolor)

    # Set mask == 1 regions to original image values
    result[mask == 1] = image[mask == 1]
    
    return result


def process_shape_detection(input_image: np.ndarray,
                            detector: ShapeDetector[T],
                            is_label: bool,
                            enable_logging: bool = False,
                            draw_mask: bool = False) -> Optional[Tuple[Dict[str, np.ndarray], T]]:
    """
    Process an image for shape detection and masking.

    Args:
        input_image (np.ndarray): Input image
        detector (ShapeDetector[T]): Shape detector instance (Ellipse or Rectangle)
        enable_logging (bool): Whether to enable logging during processing
        draw_mask (bool): Whether to create visualization image with drawn shape
        is_label (bool): If True, fill outside pixels with black (0), else white (1)

    Returns:
        Optional[Tuple[Dict[str, np.ndarray], T]]: 
            - Dictionary containing:
                - 'cut': The masked image
                - 'draw': (if draw_mask=True) Image with shape drawn
            - Shape parameters of type T
            Returns None if processing fails
    """
    if not enable_logging:
        logger_manager.disable()
    else:
        logger_manager.enable()

    try:
        # Apply circular mask if needed
        if check_corners(input_image):
            # Start a user warning if corners are white
            warnings.warn("Corner white pixels detected, applying circular mask")
            circular_mask = create_circular_mask(input_image)
            input_image = cv2.bitwise_and(input_image, circular_mask)   

        # Apply thresholding if not label image, make the edge more clear
        if is_label == False:
            origional_image = input_image.copy()
            input_image = tpi.user_threshold(input_image, 1.0/8)

        # Detect shape
        params = detector.detect(input_image)
        
        result_dict = {}

        # Cut image using shape parameters, fill outside pixels with 0, 1 if are not label
        if is_label == False:
            cut_image = origional_image.copy()
        else:
            cut_image = input_image.copy()
        fill_value = 0 if is_label else 1
        cut_image = cut_with_shape(cut_image, params, fill_value)
        result_dict['cut'] = cut_image

        if draw_mask:
            if is_label == False:
                origional_image = fb.bitconverter.binary_to_grayscale_one_image(origional_image, 'uint8')
                draw_image = cv2.cvtColor(origional_image, cv2.COLOR_GRAY2BGR)
            else:
                input_image = fb.bitconverter.binary_to_grayscale_one_image(input_image, 'uint8')
                draw_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            result_dict['draw'] = ShapeDrawer.draw_shape(draw_image, params)

        logger_manager.info(f"Detection results: {params}")
        return result_dict, params

    except Exception as e:
        logger_manager.error(f"Error processing image: {e}")
        return None