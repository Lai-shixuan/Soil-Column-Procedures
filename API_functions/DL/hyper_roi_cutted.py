import cv2
import numpy as np
import logging
import sys

from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Tuple, Optional, Protocol, TypeVar, Generic, List, Dict

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from API_functions.Soils import threshold_position_independent as tpi


# Define the new logger configuration and manager class
class LoggerManager:
    """Manages logging configuration and states"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.handler = logging.StreamHandler(sys.stdout)
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self._enabled = True

    def enable(self):
        """Enable logging"""
        self.logger.setLevel(logging.INFO)
        self._enabled = True

    def disable(self):
        """Disable logging"""
        self.logger.setLevel(logging.CRITICAL + 1)  # Set to higher than CRITICAL to disable all logging
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled"""
        return self._enabled

    @contextmanager
    def disabled(self):
        """Context manager to temporarily disable logging"""
        previous_state = self._enabled
        self.disable()
        try:
            yield
        finally:
            if previous_state:
                self.enable()

    def info(self, msg: str):
        """Log info message if enabled"""
        if self._enabled:
            self.logger.info(msg)

    def error(self, msg: str):
        """Log error message if enabled"""
        if self._enabled:
            self.logger.error(msg)

# Replace the existing logger initialization with this
logger_manager = LoggerManager()

T = TypeVar('T')  # Generic type for shape parameters

@dataclass
class ShapeParams(Protocol):
    """Base protocol for shape parameters"""
    center: Tuple[int, int]
    covered_pixels: int

@dataclass
class EllipseParams(ShapeParams):
    """Ellipse parameters"""
    long_axis: int
    short_axis: int

@dataclass
class RectangleParams(ShapeParams):
    """Rectangle parameters"""
    width: int
    height: int

class ShapeDetector(ABC, Generic[T]):
    """Abstract base class for shape detectors"""
    
    def __init__(self, min_size: int = 1):
        self.min_size = min_size
        self.translation_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    @abstractmethod
    def _create_shape_mask(self, image_shape: tuple, params: T) -> np.ndarray:
        """Create a mask for the shape"""
        pass

    def _count_covered_pixels(self, image: np.ndarray, params: T) -> int:
        """Count pixels covered by the shape"""
        mask = self._create_shape_mask(image.shape, params)
        return np.count_nonzero(cv2.bitwise_and(image, mask) == 255)

    @staticmethod
    def _get_total_target_pixels(image: np.ndarray) -> int:
        """Get total number of white pixels in image"""
        return np.count_nonzero(image == 255)

    @abstractmethod
    def detect(self, image: np.ndarray) -> T:
        """Detect the shape in the image"""
        pass

class EllipseDetector(ShapeDetector[EllipseParams]):
    """Ellipse detector implementation"""
    def __init__(self, min_size: int = 1):
        super().__init__(min_size)

    def _create_shape_mask(self, image_shape: tuple, params: EllipseParams) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.ellipse(mask, params.center, 
                   (params.long_axis // 2, params.short_axis // 2),
                   0, 0, 360, 255, -1)
        return mask

    def _try_shrink_fixed_center(self, image: np.ndarray, params: EllipseParams,
                                target_pixels: int) -> EllipseParams:
        """Try to shrink the long and short axes alternately with fixed center"""
        current_params = EllipseParams(**vars(params))
        initial_long = current_params.long_axis
        initial_short = current_params.short_axis
        
        success = True
        while success == True:
            success = False

            # Try shrinking long axis 
            if current_params.long_axis > self.min_size:
                test_params = EllipseParams(
                    center=current_params.center,
                    long_axis=current_params.long_axis - 1,
                    short_axis=current_params.short_axis,
                    covered_pixels=0
                )
                
                covered = self._count_covered_pixels(image, test_params)
                if covered == target_pixels:
                    current_params = EllipseParams(
                        center=test_params.center,
                        long_axis=test_params.long_axis,
                        short_axis=test_params.short_axis,
                        covered_pixels=covered
                    )
                    success = True
            
            # Try shrinking short axis
            if current_params.short_axis > self.min_size:
                test_params = EllipseParams(
                    center=current_params.center,
                    long_axis=current_params.long_axis,
                    short_axis=current_params.short_axis - 1,
                    covered_pixels=0
                )
                
                covered = self._count_covered_pixels(image, test_params)
                if covered == target_pixels:
                    current_params = EllipseParams(
                        center=test_params.center,
                        long_axis=test_params.long_axis,
                        short_axis=test_params.short_axis,
                        covered_pixels=covered
                    )
                    success = True
            
            # 如果已经达到最小值，退出循环
            if (current_params.long_axis <= self.min_size and 
                current_params.short_axis <= self.min_size):
                break
        
        # Only log the total shrinkage at the end if there was any
        if (initial_long != current_params.long_axis or 
            initial_short != current_params.short_axis):
            logger_manager.info(
                f"Total shrinkage at fixed center: long axis {initial_long}->{current_params.long_axis}, "
                f"short axis {initial_short}->{current_params.short_axis}"
            )
        
        return current_params
    
    def _determine_valid_directions(self, image: np.ndarray, params: EllipseParams, 
                                  target_pixels: int) -> List[Tuple[int, int]]:
        """确定有效的移动方向"""
        valid_directions = []
        
        # 只需要检查上和左两个方向
        for direction in [(-1, 0), (0, -1)]:
            test_center = (params.center[0] + direction[0], params.center[1] + direction[1])
            test_params = EllipseParams(
                center=test_center,
                long_axis=params.long_axis,
                short_axis=params.short_axis,
                covered_pixels=0
            )
            
            # 如果移动后仍能覆盖所有目标像素，则这个方向是有效的
            if self._count_covered_pixels(image, test_params) == target_pixels:
                valid_directions.append(direction)
                # 如果某个方向不可行，添加其相反方向
            else:
                valid_directions.append((-direction[0], -direction[1]))
        
        return valid_directions

    def _try_translate_and_shrink(self, image: np.ndarray, params: EllipseParams,
                                    target_pixels: int) -> EllipseParams:
        """使用新策略尝试平移并缩小椭圆"""
        current_params = EllipseParams(**vars(params))
        
        # 首先确定有效的移动方向
        valid_directions = self._determine_valid_directions(image, current_params, target_pixels)
        logger_manager.info(f"Valid movement directions: {valid_directions}")
        
        success = True
        while success:
            success = False
            
            # 对每个有效方向尝试移动和缩小
            for direction in valid_directions:
                # 尝试在这个方向上移动
                test_center = (current_params.center[0] + direction[0], 
                                current_params.center[1] + direction[1])
                test_params = EllipseParams(
                    center=test_center,
                    long_axis=current_params.long_axis,
                    short_axis=current_params.short_axis,
                    covered_pixels=0
                )
                
                # 检查移动后是否仍然覆盖所有目标像素
                if self._count_covered_pixels(image, test_params) == target_pixels:
                    # 在新位置尝试缩小
                    shrunk_params = self._try_shrink_fixed_center(image, test_params, target_pixels)
                    
                    # 如果在新位置能够获得更小的椭圆，则接受这个移动
                    if (shrunk_params.long_axis <= current_params.long_axis or 
                        shrunk_params.short_axis <= current_params.short_axis):
                        current_params = shrunk_params
                        success = True
                        logger_manager.info(
                            f"Successful move in direction {direction}, "
                            f"new axes: long={current_params.long_axis}, "
                            f"short={current_params.short_axis}"
                        )
                        break
            
            if not success:
                logger_manager.info("No further improvements possible, optimization complete")
        
        return current_params

    def detect(self, image: np.ndarray) -> EllipseParams:
        """检测最大内切椭圆"""
        height, width = image.shape
        min_side = min(height, width)
        target_pixels = self._get_total_target_pixels(image)
        
        # 初始化椭圆参数（从最大可能的椭圆开始）
        initial_params = EllipseParams(
            center=(width // 2, height // 2),
            long_axis=min_side,
            short_axis=min_side,
            covered_pixels=0
        )
        
        # 1. 固定圆心，尽可能缩小长短轴
        logger_manager.info("Step 1: Shrinking axes with fixed center...")
        current_params = self._try_shrink_fixed_center(image, initial_params, target_pixels)
        
        # 2. 尝试平移并在新位置继续缩小
        logger_manager.info("Step 2: Trying translation and shrinking...")
        final_params = self._try_translate_and_shrink(image, current_params, target_pixels)
        
        return final_params

class RectangleDetector(ShapeDetector[RectangleParams]):
    """Rectangle detector implementation"""
    
    def _create_shape_mask(self, image_shape: tuple, params: RectangleParams) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        half_width, half_height = params.width // 2, params.height // 2
        top_left = (params.center[0] - half_width, params.center[1] - half_height)
        bottom_right = (params.center[0] + half_width, params.center[1] + half_height)
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        return mask

    def _try_shrink_fixed_center(self, image: np.ndarray, params: RectangleParams,
                                target_pixels: int) -> RectangleParams:
        """在固定中心的情况下交替尝试缩小宽度和高度"""
        current_params = RectangleParams(**vars(params))
        consecutive_failures = 0
        
        while consecutive_failures < 2:
            success = False
            
            # 尝试缩小宽度
            if current_params.width > self.min_size:
                test_params = RectangleParams(
                    center=current_params.center,
                    width=current_params.width - 1,
                    height=current_params.height,
                    covered_pixels=0
                )
                
                covered = self._count_covered_pixels(image, test_params)
                if covered >= target_pixels:
                    current_params = RectangleParams(
                        center=test_params.center,
                        width=test_params.width,
                        height=test_params.height,
                        covered_pixels=covered
                    )
                    success = True
                    logger_manager.info(f"Successfully shrunk width to {current_params.width}")
            
            # 尝试缩小高度
            if current_params.height > self.min_size:
                test_params = RectangleParams(
                    center=current_params.center,
                    width=current_params.width,
                    height=current_params.height - 1,
                    covered_pixels=0
                )
                
                covered = self._count_covered_pixels(image, test_params)
                if covered >= target_pixels:
                    current_params = RectangleParams(
                        center=test_params.center,
                        width=test_params.width,
                        height=test_params.height,
                        covered_pixels=covered
                    )
                    success = True
                    logger_manager.info(f"Successfully shrunk height to {current_params.height}")
            
            # 更新连续失败计数
            if not success:
                consecutive_failures += 1
                logger_manager.info(f"No side could be shrunk, consecutive failures: {consecutive_failures}")
            else:
                consecutive_failures = 0
                logger_manager.info("Reset consecutive failures counter due to successful shrinking")
            
            # 如果已经达到最小值，退出循环
            if (current_params.width <= self.min_size and 
                current_params.height <= self.min_size):
                break
        
        return current_params
    
    def _try_translate_and_shrink(self, image: np.ndarray, params: RectangleParams,
                                target_pixels: int) -> RectangleParams:
        """尝试平移并在新位置继续缩小"""
        current_params = RectangleParams(**vars(params))
        made_improvement = True
        
        while made_improvement:
            made_improvement = False
            
            # 尝试每个平移方向
            for dx, dy in self.translation_directions:
                test_center = (current_params.center[0] + dx, current_params.center[1] + dy)
                
                # 创建一个在新中心的测试参数
                test_params = RectangleParams(
                    center=test_center,
                    width=current_params.width,
                    height=current_params.height,
                    covered_pixels=0
                )
                
                # check if we can still cover all pixels at new position
                covered = self._count_covered_pixels(image, test_params)
                if covered >= target_pixels:
                    # try shrinking at new position
                    shrunk_params = self._try_shrink_fixed_center(image, test_params, target_pixels)

                    # if we achieved better shrinkage, keep this position 
                    if (shrunk_params.width < current_params.width or 
                        shrunk_params.height < current_params.height):
                        current_params = shrunk_params
                        made_improvement = True
                        break
        
        return current_params

    def detect(self, image: np.ndarray) -> RectangleParams:
        """检测最小包围矩形"""
        # Note: image.shape returns (height, width) in NumPy order
        height, width = image.shape
        target_pixels = self._get_total_target_pixels(image)
        
        # 初始化矩形参数（从最大可能的矩形开始）
        # Note: center is in (x,y) format for OpenCV compatibility
        initial_params = RectangleParams(
            center=(width // 2, height // 2),
            width=width,
            height=height,
            covered_pixels=0
        )
        
        # 1. 固定中心，尽可能缩小宽度和高度
        logger_manager.info("Step 1: Shrinking sides with fixed center...")
        current_params = self._try_shrink_fixed_center(image, initial_params, target_pixels)
        
        # 2. 尝试平移并在新位置继续缩小
        logger_manager.info("Step 2: Trying translation and shrinking...")
        final_params = self._try_translate_and_shrink(image, current_params, target_pixels)
        
        return final_params

class ShapeDrawer:
    """Utility class for drawing shapes on images"""
    
    @staticmethod
    def draw_shape(image: np.ndarray, params: ShapeParams, color: Tuple[int, int, int] = (0, 255, 0), 
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


def _check_corners(image: np.ndarray, corner_size: int = 4) -> bool:
    """Check if all four corners contain white pixels"""
    h, w = image.shape
    corners = [
        image[0:corner_size, 0:corner_size],                    # top-left
        image[0:corner_size, w-corner_size:w],                  # top-right
        image[h-corner_size:h, 0:corner_size],                  # bottom-left
        image[h-corner_size:h, w-corner_size:w]                 # bottom-right
    ]
    return all(np.any(corner == 255) for corner in corners)

def _create_circular_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a circular mask that fits within the image
    Only used when corners contain white pixels, may because of ???
    """
    h, w = image.shape
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = np.zeros_like(image)
    mask[dist_from_center <= radius] = 255
    return mask

def apply_shape_mask(image: np.ndarray, params: ShapeParams) -> np.ndarray:
    """Create mask from shape parameters and apply it to image"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create single channel mask
    
    if isinstance(params, EllipseParams):
        cv2.ellipse(mask, params.center, 
                    (params.long_axis // 2, params.short_axis // 2),
                    0, 0, 360, 255, -1)
    elif isinstance(params, RectangleParams):
        half_width, half_height = params.width // 2, params.height // 2
        top_left = (params.center[0] - half_width, params.center[1] - half_height)
        bottom_right = (params.center[0] + half_width, params.center[1] + half_height)
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    
    # Handle both color and grayscale images
    if len(image.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return cv2.bitwise_and(image, mask)

def cut_with_shape(image: np.ndarray, params: ShapeParams, fillcolor: int) -> np.ndarray:
    """
    Apply detected shape parameters to cut another grayscale image and fill outside pixels with 255
    All parametes shrink the shape by 2 pixels to avoid edge artifacts

    Args:
        image (np.ndarray): Input grayscale image to be cut
        params (ShapeParams): Shape parameters (EllipseParams or RectangleParams)

    Returns:
        np.ndarray: Cut grayscale image with outside pixels set to 255
        
    Raises:
        ValueError: If input image is not grayscale
    """

    shrinked_params = 2

    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (single channel)")
    
    # Create mask matching the input image dimensions
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    # Draw the shape in color (255) on the mask
    if isinstance(params, EllipseParams):
        cv2.ellipse(mask, params.center, 
                    ((params.long_axis - shrinked_params) // 2, (params.short_axis - shrinked_params) // 2),
                    0, 0, 360, 255, -1)
    elif isinstance(params, RectangleParams):
        half_width, half_height = (params.width - shrinked_params) // 2, (params.height - shrinked_params) // 2
        top_left = (params.center[0] - half_width, params.center[1] - half_height)
        bottom_right = (params.center[0] + half_width, params.center[1] + half_height)
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    
    # Create output image (all white)
    result = np.full_like(image, fillcolor)

    # Set mask == 255 regions to original image values
    result[mask == 255] = image[mask == 255]
    
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
        is_label (bool): If True, fill outside pixels with black (0), else white (255)

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
        if _check_corners(input_image):
            # Start a user warning if corners are white
            Warning.warn("Corner white pixels detected, applying circular mask")
            circular_mask = _create_circular_mask(input_image)
            input_image = cv2.bitwise_and(input_image, circular_mask)   

        # Apply thresholding if not label image, make the edge more clear
        if is_label == False:
            origional_image = input_image.copy()
            input_image = tpi.user_threshold(input_image, 255//8)

        # Detect shape
        params = detector.detect(input_image)
        
        result_dict = {}

        # Cut image using shape parameters, fill outside pixels with 0, 255 if are not label
        if is_label == False:
            cut_image = origional_image.copy()
        else:
            cut_image = input_image.copy()
        fill_value = 0 if is_label else 255
        cut_image = cut_with_shape(cut_image, params, fill_value)
        result_dict['cut'] = cut_image

        if draw_mask:
            if is_label == False:
                draw_image = cv2.cvtColor(origional_image, cv2.COLOR_GRAY2BGR)
            else:
                draw_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            result_dict['draw'] = ShapeDrawer.draw_shape(draw_image, params)

        logger_manager.info(f"Detection results: {params}")
        return result_dict, params

    except Exception as e:
        logger_manager.error(f"Error processing image: {e}")
        return None