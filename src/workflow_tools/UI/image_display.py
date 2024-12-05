import cv2
import os
import numpy as np

from matplotlib.axes import Axes
from typing import List, Optional
from .image_utils import resize_image
from ...API_functions.DL.multi_input_adapter import harmonized_normalize
from ...API_functions.Soils.threshold_position_independent import user_threshold

class ImageDisplay:
    """Handles the display of image pairs with zoom and threshold functionality.
    
    This class manages the matplotlib axes for displaying image pairs, including:
        - Image resizing and processing
        - Zoom region handling
        - Threshold visualization
        - Axes layout and visibility
    
    Attributes:
        ax (List[Axes]): List of matplotlib axes for image display
        target_width (int): Target width for displayed images
        target_height (int): Target height for displayed images
    """

    def __init__(self, axes: List[Axes], target_width: int, target_height: int):
        """Initialize the ImageDisplay instance.

        Args:
            axes: List of matplotlib axes for image display (2x3 grid)
            target_width: Width to resize images to
            target_height: Height to resize images to
        """
        self.ax = axes
        self.target_width = target_width
        self.target_height = target_height
        self.current_threshold_result = None  # Add this line

    def display_pair(self, img1_path: str, img2_path: str, zoom_region=None, threshold_enabled=False, threshold_value=None) -> bool:
        """Display a pair of images with optional zoom and threshold.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            zoom_region: Optional ZoomRegion specifying area to zoom
            threshold_enabled: Whether thresholding is enabled
            threshold_value: Threshold value between 0 and 1

        Returns:
            bool: True if display successful, False otherwise
        """
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

            if img1 is None or img2 is None:
                return False

            # Process images: harmonize first
            img1_norm = harmonized_normalize(img1)
            img2_norm = harmonized_normalize(img2)

            # Apply threshold before resize if enabled
            if threshold_enabled and threshold_value is not None:
                mask = (img1_norm > 0) & (img1_norm < 1)
                thresh_result = user_threshold(img1_norm, threshold_value, mask)
                self.current_threshold_result = thresh_result  # Store the result
                thresh_resized = resize_image(thresh_result, self.target_width, self.target_height)
            else:
                self.current_threshold_result = None

            # Resize normalized images
            img1_display = resize_image(img1_norm, self.target_width, self.target_height)
            img2_display = resize_image(img2_norm, self.target_width, self.target_height)

            # Clear and set images
            for ax in self.ax:
                ax.clear()

            self.ax[0].imshow(img1_display, cmap='gray')
            self.ax[1].imshow(img2_display, cmap='gray')

            if zoom_region:
                self._handle_zoom(img1_norm, img2_norm, img1_display, img2_display, 
                                thresh_result if threshold_enabled else None,
                                zoom_region, threshold_enabled, threshold_value)
            else:
                self._handle_normal_view(img1_display, img2_display, 
                                       thresh_resized if threshold_enabled else None,
                                       threshold_enabled)

            # Set titles for all visible axes
            base1 = os.path.basename(img1_path)
            base2 = os.path.basename(img2_path)
            self.ax[0].set_title(f"Image 1: {base1}")
            self.ax[1].set_title(f"Image 2: {base2}")
            if threshold_enabled:
                self.ax[2].set_title("Threshold Result")
                if zoom_region:
                    self.ax[3].set_title("Image 1 (Zoomed)")
                    self.ax[4].set_title("Image 2 (Zoomed)")
                    self.ax[5].set_title("Threshold (Zoomed)")
            elif zoom_region:
                self.ax[3].set_title("Image 1 (Zoomed)")
                self.ax[4].set_title("Image 2 (Zoomed)")

            # Set axes properties
            for ax in self.ax:
                if ax.get_visible():
                    ax.axis('off')

            return True
        except Exception as e:
            print(f"Error displaying images: {e}")
            return False

    def _handle_zoom(self, img1_norm, img2_norm, img1_display, img2_display, thresh_full, 
                    zoom_region, threshold_enabled, threshold_value):
        """Handle zoomed view display."""
        # Create deep copies for marking
        img1_marked = img1_display.copy()
        img2_marked = img2_display.copy()
        
        # Get zoom coordinates and bounds
        y, x = zoom_region.y, zoom_region.x
        h, w = zoom_region.height, zoom_region.width
        
        height1, width1 = img1_display.shape[:2]
        y1 = max(0, min(y, height1 - 1))
        x1 = max(0, min(x, width1 - 1))
        y2_1 = max(0, min(y + h, height1))
        x2_1 = max(0, min(x + w, width1))
        
        # Draw rectangles
        cv2.rectangle(img1_marked, (x1, y1), (x2_1, y2_1), 1.0, 2)
        cv2.rectangle(img2_marked, (x1, y1), (x2_1, y2_1), 1.0, 2)
        
        # Extract and resize zoomed regions
        zoom_scale = img1_norm.shape[0] / img1_display.shape[0]
        norm_x1, norm_y1 = int(x1 * zoom_scale), int(y1 * zoom_scale)
        norm_x2, norm_y2 = int(x2_1 * zoom_scale), int(y2_1 * zoom_scale)
        
        img1_zoomed = resize_image(img1_norm[norm_y1:norm_y2, norm_x1:norm_x2], 
                                 self.target_width, self.target_height)
        img2_zoomed = resize_image(img2_norm[norm_y1:norm_y2, norm_x1:norm_x2], 
                                 self.target_width, self.target_height)
        
        # Display images
        self.ax[0].imshow(img1_marked, cmap='gray')
        self.ax[1].imshow(img2_marked, cmap='gray')
        
        self.ax[3].set_visible(True)
        self.ax[4].set_visible(True)
        self.ax[3].imshow(img1_zoomed, cmap='gray')
        self.ax[4].imshow(img2_zoomed, cmap='gray')
        
        if threshold_enabled and thresh_full is not None:
            self.ax[2].set_visible(True)
            thresh_marked = resize_image(thresh_full, self.target_width, self.target_height)
            cv2.rectangle(thresh_marked, (x1, y1), (x2_1, y2_1), 1.0, 2)
            self.ax[2].imshow(thresh_marked, cmap='gray')
            
            thresh_zoomed = resize_image(thresh_full[norm_y1:norm_y2, norm_x1:norm_x2], 
                                       self.target_width, self.target_height)
            self.ax[5].set_visible(True)
            self.ax[5].imshow(thresh_zoomed, cmap='gray')
        else:
            self.ax[2].set_visible(False)
            self.ax[5].set_visible(False)

    def _handle_normal_view(self, img1_display, img2_display, thresh_resized, threshold_enabled):
        """Handle normal view display."""
        self.ax[0].imshow(img1_display, cmap='gray')
        self.ax[1].imshow(img2_display, cmap='gray')
        
        if threshold_enabled and thresh_resized is not None:
            self.ax[2].set_visible(True)
            self.ax[2].imshow(thresh_resized, cmap='gray')
        else:
            self.ax[2].set_visible(False)
        
        # Hide bottom row
        for i in range(3, 6):
            self.ax[i].set_visible(False)

    def _apply_threshold(self, img: np.ndarray, threshold: float, ax_index: Optional[int] = None) -> np.ndarray:
        """Apply threshold to image and optionally display in specified axis."""
        mask = (img > 0) & (img < 1)
        result = user_threshold(img, threshold, mask)
        if ax_index is not None:
            self.ax[ax_index].set_visible(True)
            self.ax[ax_index].imshow(result, cmap='gray')
        return result