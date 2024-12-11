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
        self.mean_filter_enabled = False
        self.filter_size = 3
        self.filtered_result = None
        self.histogram_eq_enabled = False

    def _get_shared_color_limits(self, img1, img2=None):
        """Calculate shared color limits for a pair of images."""
        vmin = img1.min()
        vmax = img1.max()
        
        if img2 is not None:
            vmin = min(vmin, img2.min())
            vmax = max(vmax, img2.max())
            
        return vmin, vmax

    def display_pair(self, img1_path: str, img2_path: str = None, zoom_region=None, threshold_enabled=False, threshold_value=None) -> bool:
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED) if img2_path else None

            if img1 is None:
                return False

            # Process first image
            img1_norm = harmonized_normalize(img1)
            img1_display = resize_image(img1_norm, self.target_width, self.target_height)

            # Process second image if available
            if img2 is not None:
                img2_norm = harmonized_normalize(img2)
                img2_display = resize_image(img2_norm, self.target_width, self.target_height)
            else:
                img2_norm = None
                img2_display = None

            # Store original image for processing
            img1_for_processing = img1_norm.copy()
            thresh_result = None
            
            # Apply filter and threshold if enabled
            if threshold_enabled and threshold_value is not None:
                # First apply mean filter if enabled
                if self.mean_filter_enabled and self.filter_size >= 3:
                    img1_for_processing = self.apply_mean_filter(img1_for_processing, self.filter_size)
                
                if self.histogram_eq_enabled:
                    img1_for_processing = self.apply_histogram_equalization(img1_for_processing)

                # Then apply threshold
                mask = (img1_for_processing > 0) & (img1_for_processing < 1)
                thresh_result = user_threshold(img1_for_processing, threshold_value, mask)
                self.current_threshold_result = thresh_result  # Store for saving
                thresh_resized = resize_image(thresh_result, self.target_width, self.target_height)
            else:
                self.current_threshold_result = None
                thresh_resized = None

            # Calculate shared color limits for each pair
            main_limits = self._get_shared_color_limits(img1_display, img2_display)
            
            # Clear all axes
            for ax in self.ax:
                ax.clear()
                ax.set_visible(False)

            # Display first image
            self.ax[0].set_visible(True)
            self.ax[0].imshow(img1_display, cmap='gray', vmin=main_limits[0], vmax=main_limits[1])
            self.ax[0].set_title(f"Image 1: {os.path.basename(img1_path)}")

            # Display second image if available
            if img2_display is not None:
                self.ax[1].set_visible(True)
                self.ax[1].imshow(img2_display, cmap='gray', vmin=main_limits[0], vmax=main_limits[1])
                self.ax[1].set_title(f"Image 2: {os.path.basename(img2_path)}")

            # Handle threshold display with its own color limits
            if threshold_enabled and thresh_resized is not None:
                self.ax[2].set_visible(True)
                # Always show the thresholded result in column 3
                self.ax[2].imshow(thresh_resized, cmap='gray', vmin=0, vmax=1)
                title = "Threshold Result"
                if self.mean_filter_enabled:
                    title += " (Filtered)"
                if self.histogram_eq_enabled:
                    title += " (Equalized)"
                self.ax[2].set_title(title)

            # Handle zoom regions
            if zoom_region:
                self._handle_zoom(img1_norm, img2_norm, img1_display, img2_display, 
                                thresh_result if threshold_enabled else None,
                                zoom_region, threshold_enabled, threshold_value,
                                main_limits)  # Pass the color limits

            # Set axes properties for visible axes
            for ax in self.ax:
                if ax.get_visible():
                    ax.axis('off')

            return True
        except Exception as e:
            print(f"Error displaying images: {e}")
            return False

    def _handle_zoom(self, img1_norm, img2_norm, img1_display, img2_display, thresh_full, 
                    zoom_region, threshold_enabled, threshold_value, main_limits):
        """Handle zoomed view display."""
        # Create deep copies for marking
        img1_marked = img1_display.copy()
        
        # Get zoom coordinates and bounds
        y, x = zoom_region.y, zoom_region.x
        h, w = zoom_region.height, zoom_region.width
        
        height1, width1 = img1_display.shape[:2]
        y1 = max(0, min(y, height1 - 1))
        x1 = max(0, min(x, width1 - 1))
        y2_1 = max(0, min(y + h, height1))
        x2_1 = max(0, min(x + w, width1))
        
        # Draw rectangles on copies, not originals
        cv2.rectangle(img1_marked, (x1, y1), (x2_1, y2_1), 1.0, 2)
        
        # Calculate zoom scale
        zoom_scale = img1_norm.shape[0] / img1_display.shape[0]
        norm_x1 = int(x1 * zoom_scale)
        norm_y1 = int(y1 * zoom_scale)
        norm_x2 = int(x2_1 * zoom_scale)
        norm_y2 = int(y2_1 * zoom_scale)
        
        # Extract and resize zoomed regions
        if norm_y2 > norm_y1 and norm_x2 > norm_x1:
            img1_zoomed = resize_image(img1_norm[norm_y1:norm_y2, norm_x1:norm_x2], 
                                     self.target_width, self.target_height)
        else:
            img1_zoomed = img1_display  # Fallback to full image if zoom region is invalid
        
        # Display first image and its zoom using shared limits
        self.ax[0].imshow(img1_marked, cmap='gray', vmin=main_limits[0], vmax=main_limits[1])
        self.ax[3].set_visible(True)
        self.ax[3].imshow(img1_zoomed, cmap='gray', vmin=main_limits[0], vmax=main_limits[1])
        
        # Handle second image if available
        if img2_display is not None and img2_norm is not None:
            img2_marked = img2_display.copy()
            cv2.rectangle(img2_marked, (x1, y1), (x2_1, y2_1), 1.0, 2)
            
            if norm_y2 > norm_y1 and norm_x2 > norm_x1:
                img2_zoomed = resize_image(img2_norm[norm_y1:norm_y2, norm_x1:norm_x2], 
                                         self.target_width, self.target_height)
            else:
                img2_zoomed = img2_display
                
            self.ax[1].imshow(img2_marked, cmap='gray', vmin=main_limits[0], vmax=main_limits[1])
            self.ax[4].set_visible(True)
            self.ax[4].imshow(img2_zoomed, cmap='gray', vmin=main_limits[0], vmax=main_limits[1])
        else:
            self.ax[1].set_visible(False)
            self.ax[4].set_visible(False)
        
        # Handle threshold display
        if threshold_enabled and thresh_full is not None:
            # Use thresh_full directly as it already contains the filtered+threshold result
            thresh_marked = resize_image(thresh_full, self.target_width, self.target_height)
            cv2.rectangle(thresh_marked, (x1, y1), (x2_1, y2_1), 1.0, 2)
            self.ax[2].set_visible(True)
            # 使用固定的显示范围0-1
            self.ax[2].imshow(thresh_marked, cmap='gray', vmin=0, vmax=1)
            
            if norm_y2 > norm_y1 and norm_x2 > norm_x1:
                thresh_zoomed = resize_image(thresh_full[norm_y1:norm_y2, norm_x1:norm_x2], 
                                           self.target_width, self.target_height)
                self.ax[5].set_visible(True)
                # 对阈值的缩放区域也使用相同的显示范围
                self.ax[5].imshow(thresh_zoomed, cmap='gray', vmin=0, vmax=1)
            else:
                self.ax[5].set_visible(False)
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

    def apply_mean_filter(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply mean filter to the image with given kernel size."""
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        return cv2.blur(image, (kernel_size, kernel_size))

    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to the image."""
        # Convert to uint8 for histogram equalization
        img_uint8 = (image * 255).astype(np.uint8)
        equalized = cv2.equalizeHist(img_uint8)
        # Convert back to float32 and normalize to 0-1
        return equalized.astype(np.float32) / 255.0