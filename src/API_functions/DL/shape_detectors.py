import cv2
import numpy as np
from typing import List, Tuple

from .shape_base import ShapeDetector, EllipseParams, RectangleParams
from .log import logger_manager


class EllipseDetector(ShapeDetector[EllipseParams]):
    """Ellipse detector implementation"""
    def __init__(self, min_size: int = 1):
        super().__init__(min_size)

    def _create_shape_mask(self, image_shape: tuple, params: EllipseParams) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.float32)
        cv2.ellipse(mask, params.center, 
                   (params.long_axis // 2, params.short_axis // 2),
                    0, 0, 360, 1, -1)
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
        mask = np.zeros(image_shape, dtype=np.float32)
        half_width, half_height = params.width // 2, params.height // 2
        top_left = (params.center[0] - half_width, params.center[1] - half_height)
        bottom_right = (params.center[0] + half_width, params.center[1] + half_height)
        cv2.rectangle(mask, top_left, bottom_right, 1, -1)
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
        
        # 2. 尝试平移并在新���置继续缩小
        logger_manager.info("Step 2: Trying translation and shrinking...")
        final_params = self._try_translate_and_shrink(image, current_params, target_pixels)
        
        return final_params