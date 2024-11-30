import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, TypeVar, Generic, Protocol


@dataclass
class ShapeParams(Protocol):
    """Base protocol for shape parameters"""
    center: Tuple[int, int]
    covered_pixels: int

T = TypeVar('T')  # Generic type for shape parameters

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