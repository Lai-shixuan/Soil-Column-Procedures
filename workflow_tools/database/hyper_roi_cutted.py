#%%
# Import the required libraries

from abc import ABC, abstractmethod
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Protocol, TypeVar, Generic, List, Dict
import logging
from pathlib import Path
from contextlib import contextmanager
from tqdm import tqdm
import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from API_functions import file_batch as fb

#%%
# Define the new logger configuration and manager class

# Replace the existing logger configuration with this new class
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
    """Create a circular mask that fits within the image"""
    h, w = image.shape
    center = (w // 2, h // 2)
    radius = min(w, h) // 2
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = np.zeros_like(image)
    mask[dist_from_center <= radius] = 255
    return mask

# Add this new utility function for image cutting
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

def process_shape_detection(binary_input_path: str, second_input_path: str,
                          output_dir: Optional[str], detector: ShapeDetector[T],
                          enable_logging: bool = True) -> Optional[Tuple[Dict[str, np.ndarray], T]]:
    """Enhanced shape detection processor returning processed images and parameters"""
    if not enable_logging:
        logger_manager.disable()
    else:
        logger_manager.enable()

    try:
        # Convert paths to Path objects
        binary_input_path = Path(binary_input_path)
        second_input_path = Path(second_input_path)
        
        # Validate inputs
        if not binary_input_path.exists():
            raise FileNotFoundError(f"Binary input file not found: {binary_input_path}")
        if not second_input_path.exists():
            raise FileNotFoundError(f"Second input file not found: {second_input_path}")
        
        # Read both images in grayscale
        binary_image = cv2.imread(str(binary_input_path), cv2.IMREAD_GRAYSCALE)
        second_image = cv2.imread(str(second_input_path), cv2.IMREAD_GRAYSCALE)
        if binary_image is None or second_image is None:
            raise ValueError("Failed to load one or both images")

        # Apply circular mask if needed
        if _check_corners(binary_image):
            logger_manager.info("Corner white pixels detected, applying circular mask")
            circular_mask = _create_circular_mask(binary_image)
            binary_image = cv2.bitwise_and(binary_image, circular_mask)

        # Detect shape
        params = detector.detect(binary_image)
        
        processed_images = {
            'label_draw': None,
            'image_draw': None,
            'label_cut': None,
            'image_cut': None
        }

        # Convert to color for drawing
        binary_result = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        second_result = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)
        
        # Draw shapes
        processed_images['label_draw'] = ShapeDrawer.draw_shape(binary_result, params)
        processed_images['image_draw'] = ShapeDrawer.draw_shape(second_result, params)
        
        # Create masked images (cuts) without drawing
        processed_images['label_cut'] = apply_shape_mask(binary_image, params)
        
        # Special handling for second image cut: set outside pixels to 255
        shape_mask = np.zeros(second_image.shape, dtype=np.uint8)
        if isinstance(params, EllipseParams):
            cv2.ellipse(shape_mask, params.center, 
                       (params.long_axis // 2, params.short_axis // 2),
                       0, 0, 360, 255, -1)
        # Create cut image with outside pixels as 255
        cut_image = second_image.copy()
        cut_image[shape_mask == 0] = 255  # Set outside pixels to white
        cut_image[shape_mask == 255] = second_image[shape_mask == 255]  # Keep original pixels inside
        processed_images['image_cut'] = cut_image

        # Save all images if output_dir is provided
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            base_name = binary_input_path.stem
            
            cv2.imwrite(str(out_path / f"{base_name}_label-draw.png"), processed_images['label_draw'])
            cv2.imwrite(str(out_path / f"{base_name}_image-draw.png"), processed_images['image_draw'])
            cv2.imwrite(str(out_path / f"{base_name}_label-cut.png"), processed_images['label_cut'])
            cv2.imwrite(str(out_path / f"{base_name}_image-cut.png"), processed_images['image_cut'])
            
            logger_manager.info(f"All results saved to: {out_path}")

        logger_manager.info(f"Detection results: {params}")
        return processed_images, params

    except Exception as e:
        logger_manager.error(f"Error processing images: {e}")
        return None

def batch_process_images(label_paths: List[str], image_paths: List[str], 
                        output_base_dir: str, detector: ShapeDetector[T],
                        enable_logging: bool = True) -> None:
    """
    Process multiple pairs of images and organize outputs into separate folders.
    """
    if len(label_paths) != len(image_paths):
        raise ValueError("Number of label and image paths must match")

    # Create output directories
    output_base_dir = Path(output_base_dir)
    output_dirs = {
        'label_draw': output_base_dir / 'label_draw',
        'image_draw': output_base_dir / 'image_draw',
        'label_cut': output_base_dir / 'label_cut',
        'image_cut': output_base_dir / 'image_cut'
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Set logging state once for all processing
    if not enable_logging:
        logger_manager.disable()
    
    try:
        for label_path, image_path in tqdm(zip(label_paths, image_paths), total=len(label_paths)):
            try:
                # Process the image pair
                result = process_shape_detection(
                    label_path, image_path, None, detector, False)
                
                if result is None:
                    continue
                    
                processed_images, params = result
                base_name = Path(label_path).stem
                
                # Save images to their respective folders
                for img_type, img in processed_images.items():
                    output_path = output_dirs[img_type] / f"{base_name}.png"
                    cv2.imwrite(str(output_path), img)
                
                if enable_logging:
                    logger_manager.info(f"Processed {base_name} - Shape params: {params}")
                    
            except Exception as e:
                if enable_logging:
                    logger_manager.error(f"Error processing {label_path}: {e}")
                continue
    finally:
        # Restore logging state
        if not enable_logging:
            logger_manager.enable()

# Update the example usage
if __name__ == "__main__":
    # Single image processing
    # binary_input = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/0.Origin/Origin-special_images_154/1.tryhard/CG_P2_O_a_cut_nlm_zcor_seg_z737.jpg'
    # second_input = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/0.Origin/Origin-special_images_154/1.tryhard/CG_P2_O_a_cut_nlm_zcor_z737.jpg'
    # output_dir = "./results/"
    # process_shape_detection(binary_input, second_input, output_dir, EllipseDetector())

    # Batch processing
    label_folder = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/labels/'
    image_folder = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/images/'
    
    # Get all matching files from both folders

    label_paths = fb.get_image_names(label_folder, None, 'png')
    image_paths = fb.get_image_names(image_folder, None, 'png')
    
    batch_process_images(label_paths, image_paths, "./batch_results/", EllipseDetector(), enable_logging=False)
