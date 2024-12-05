from typing import List, Optional, Tuple
import sys
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QHBoxLayout, QMessageBox,
    QScrollBar, QLabel, QLineEdit, QCheckBox  # Add this import
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use('Qt5Agg')

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")
from src.API_functions.Images.file_batch import get_image_names
from src.API_functions.Visual.file_compare import ZoomRegion, zoom_in
from src.API_functions.Soils.threshold_position_independent import user_threshold  # Add this import

class ImageViewer(QMainWindow):
    """A PyQt5-based image viewer for comparing images from two folders.

    This viewer allows users to:
        - Select two folders containing images
        - Navigate through corresponding images
        - Zoom into specific regions of the images
        - Compare images side by side

    Attributes:
        folder1 (str): Path to the first folder containing images
        folder2 (str): Path to the second folder containing images
        images1_list (List[str]): List of image paths from folder1
        images2_list (List[str]): List of image paths from folder2
        zoom_region (Optional[ZoomRegion]): Current zoom region settings
    """
    TARGET_WIDTH = 512  # Fixed width for display
    TARGET_HEIGHT = 512  # Fixed height for display
    WINDOW_WIDTH = 1600  # 16:9 ratio
    WINDOW_HEIGHT = 900

    def __init__(self):
        """Initialize the ImageViewer with UI components and event handlers."""
        super().__init__()
        
        self.folder1 = None
        self.folder2 = None
        self.images1_list = []
        self.images2_list = []
        
        # Index of the current image
        self.index1 = 0
        self.index2 = 0
        
        # Create the Qt5 main window
        self.setWindowTitle('Image Viewer')
        
        # 修改主窗口大小和比例
        self.setMinimumSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | 
                          Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        # 创建主布局为水平布局
        main_layout = QHBoxLayout()
        
        # 创建左侧控制面板
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignTop)
        
        # 创建所有控件
        self.btn_folder1 = QPushButton('Select Folder 1')
        self.btn_folder2 = QPushButton('Select Folder 2')
        
        # 创建标签 - 移到这里
        self.label_folder1 = QLabel('No folder selected')
        self.label_folder2 = QLabel('No folder selected')

        # Connect buttons to functions
        self.btn_folder1.clicked.connect(lambda: self.select_folder(1))
        self.btn_folder2.clicked.connect(lambda: self.select_folder(2))

        button_group = QVBoxLayout()
        button_group.addWidget(self.btn_folder1)
        button_group.addWidget(self.label_folder1)
        button_group.addWidget(self.btn_folder2)
        button_group.addWidget(self.label_folder2)
        button_group.addStretch()
        
        # 缩放控制
        zoom_group = QVBoxLayout()
        zoom_group.setSpacing(5)
        
        # Create zoom checkbox and input layout
        zoom_header = QHBoxLayout()
        self.enable_zoom = QCheckBox('Enable Zoom')
        self.enable_zoom.setEnabled(False)  # Disable initially
        self.enable_zoom.stateChanged.connect(self.on_zoom_enabled)
        zoom_header.addWidget(self.enable_zoom)
        zoom_group.addLayout(zoom_header)
        
        # 创建网格布局用于缩放控制
        zoom_grid = QHBoxLayout()
        zoom_labels = QVBoxLayout()
        zoom_inputs = QVBoxLayout()
        
        for label in ['Y:', 'X:', 'Width:', 'Height:']:
            lbl = QLabel(label)
            lbl.setFixedWidth(50)
            zoom_labels.addWidget(lbl)
        
        self.zoom_y = QLineEdit()
        self.zoom_x = QLineEdit()
        self.zoom_width = QLineEdit()
        self.zoom_height = QLineEdit()
        
        for widget in [self.zoom_y, self.zoom_x, self.zoom_width, self.zoom_height]:
            widget.setFixedWidth(60)
            widget.setEnabled(False)  # Disable initially
            widget.textChanged.connect(self.on_zoom_values_changed)
            zoom_inputs.addWidget(widget)
        
        zoom_grid.addLayout(zoom_labels)
        zoom_grid.addLayout(zoom_inputs)
        zoom_group.addLayout(zoom_grid)
        zoom_group.addStretch()

        # Remove old zoom buttons and their connections
        
        # Add threshold control with value input
        threshold_group = QVBoxLayout()
        threshold_header = QHBoxLayout()
        self.enable_threshold = QCheckBox('Enable Threshold')
        self.enable_threshold.setEnabled(False)  # Disable initially
        self.enable_threshold.stateChanged.connect(self.on_threshold_enabled)  # Add this line
        self.threshold_value = QLineEdit()
        self.threshold_value.setFixedWidth(60)
        self.threshold_value.setPlaceholderText('0-1')
        self.threshold_value.setEnabled(False)  # Disable initially
        self.threshold_value.textChanged.connect(self.on_threshold_value_changed)
        threshold_header.addWidget(self.enable_threshold)
        threshold_header.addWidget(self.threshold_value)
        threshold_group.addLayout(threshold_header)
        threshold_group.addStretch()

        # 将控件组添加到左侧面板
        left_panel.addLayout(button_group)
        left_panel.addLayout(zoom_group)
        left_panel.addLayout(threshold_group)  # Add threshold group

        # 创建中央图像显示区域
        central_layout = QHBoxLayout()
        # Always create a 2x3 grid
        self.fig, self.ax = plt.subplots(2, 3, figsize=(20, 9))
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                                wspace=0.2, hspace=0.2)
        self.ax = self.ax.flatten()
        self.canvas = FigureCanvas(self.fig)
        central_layout.addWidget(self.canvas)
        
        # 创建右侧滚动条面板
        right_panel = QVBoxLayout()
        self.scrollbar = QScrollBar()
        self.scrollbar.setOrientation(Qt.Vertical)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(0)  # Will be updated when images are loaded
        self.scrollbar.valueChanged.connect(self.scroll_images)
        right_panel.addWidget(self.scrollbar)
        
        # 将所有面板添加到主布局
        main_layout.addLayout(left_panel, stretch=1)  # 左侧面板占比小
        main_layout.addLayout(central_layout, stretch=8)  # 中央区域占比大
        main_layout.addLayout(right_panel, stretch=1)  # 右侧面板占比小
        
        # 创建主窗口部件并设置布局
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        self.pattern_length = None  # Store the common prefix length between folders

        # Initialize zoom region
        self.zoom_region = None
        
        # Remove these lines as we no longer have zoom buttons
        # Connect zoom buttons
        # self.zoom_button.clicked.connect(self.apply_zoom)
        # self.reset_zoom_button.clicked.connect(self.reset_zoom)

    def get_image_files(self, folder: str) -> List[str]:
        """Get a sorted list of TIF image files from the specified folder.

        Args:
            folder: Path to the folder containing images.

        Returns:
            List of image filenames without extensions.
        """
        files = get_image_names(folder, None, 'tif')
        return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])

    def get_common_prefix_length(self, str1: str, str2: str) -> int:
        """Find the length of common prefix between two strings.

        Args:
            str1: First string to compare
            str2: Second string to compare

        Returns:
            Length of the common prefix
        """
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return i
        return min_len

    def analyze_filename_pattern(self, files1: List[str], files2: List[str]) -> Optional[int]:
        """Analyze file pairs to determine consistent naming pattern length.

        Args:
            files1: List of files from first folder
            files2: List of files from second folder

        Returns:
            Length of common prefix if consistent pattern exists, None otherwise
        """
        if len(files1) != len(files2):
            return None
            
        sample_size = min(5, len(files1))
        lengths = []
        
        # Compare corresponding files from both folders
        for i in range(sample_size):
            name1 = os.path.splitext(os.path.basename(files1[i]))[0]
            name2 = os.path.splitext(os.path.basename(files2[i]))[0]
            common_len = self.get_common_prefix_length(name1, name2)
            lengths.append(common_len)
            
        # All pairs should have the same prefix length
        if len(set(lengths)) != 1:
            return None
            
        return lengths[0]

    def validate_folders(self) -> bool:
        """Validate that both folders contain matching image files.

        Checks:
            - Both folders exist and contain images
            - Equal number of files
            - Consistent naming patterns

        Returns:
            True if folders are valid, False otherwise
        """
        if not self.folder1 or not self.folder2:
            return False
            
        files1 = sorted(get_image_names(self.folder1, None, 'tif'))
        files2 = sorted(get_image_names(self.folder2, None, 'tif'))
        
        if not files1 or not files2:
            QMessageBox.warning(self, 'Warning', 'One or both folders are empty!')
            return False
            
        if len(files1) != len(files2):
            QMessageBox.warning(self, 'Warning', 'Folders contain different numbers of files!')
            return False

        # Analyze pattern from first 5 file pairs
        self.pattern_length = self.analyze_filename_pattern(files1, files2)
        if self.pattern_length is None:
            QMessageBox.warning(self, 'Warning', 
                              'Files do not follow a consistent naming pattern between folders!')
            return False

        # Validate all remaining files follow the same pattern
        for f1, f2 in zip(files1, files2):
            name1 = os.path.splitext(os.path.basename(f1))[0]
            name2 = os.path.splitext(os.path.basename(f2))[0]
            if name1[:self.pattern_length] != name2[:self.pattern_length]:
                QMessageBox.warning(self, 'Warning',
                                  f'Pattern mismatch: {name1} vs {name2}')
                return False

        self.images1_list = files1
        self.images2_list = files2
        # Update scrollbar range
        self.scrollbar.setMaximum(len(files1) - 1)

        if result := (self.images1_list and self.images2_list):
            # Enable controls only when both folders are valid
            self.enable_controls(True)
            # Start viewing immediately when folders are validated
            self.scrollbar.setValue(0)
            self.display_images()
            self.canvas.mpl_connect('scroll_event', self.on_scroll)
        else:
            # Disable controls if validation fails
            self.enable_controls(False)
            
        return result

    def enable_controls(self, enabled: bool) -> None:
        """Enable or disable all controls that require folders to be selected."""
        # Enable/disable zoom controls - only enable checkbox
        self.enable_zoom.setEnabled(enabled)
        # Zoom inputs remain disabled until checkbox is checked
        zoom_enabled = enabled and self.enable_zoom.isChecked()
        self.zoom_y.setEnabled(zoom_enabled)
        self.zoom_x.setEnabled(zoom_enabled)
        self.zoom_width.setEnabled(zoom_enabled)
        self.zoom_height.setEnabled(zoom_enabled)
        
        # Enable/disable threshold controls
        self.enable_threshold.setEnabled(enabled)
        self.threshold_value.setEnabled(enabled and self.enable_threshold.isChecked())

    def select_folder(self, folder_num: int) -> None:
        """Open file dialog to select an image folder.

        Args:
            folder_num: 1 for first folder, 2 for second folder
        """
        folder = QFileDialog.getExistingDirectory(self, f'Select Folder {folder_num}')
        if folder:
            if folder_num == 1:
                self.folder1 = folder
                self.images1_list = get_image_names(folder, None, 'tif')
                self.images1_list.sort()
                self.label_folder1.setText(f"Folder 1: {folder}")  # Update label
            else:
                self.folder2 = folder
                self.images2_list = get_image_names(folder, None, 'tif')
            
            # Validate and show images if both folders are selected
            if self.folder1 and self.folder2:
                self.validate_folders()
            else:
                self.enable_controls(False)

    def apply_zoom(self) -> None:
        """Apply zoom settings from input fields to the images."""
        try:
            y = int(self.zoom_y.text() or 0)
            x = int(self.zoom_x.text() or 0)
            width = int(self.zoom_width.text() or 0)
            height = int(self.zoom_height.text() or 0)
            
            if width > 0 and height > 0:
                self.zoom_region = ZoomRegion(y, x, width, height)
                self.display_images()
        except ValueError:
            QMessageBox.warning(self, 'Warning', 'Please enter valid numeric values for zoom region.')

    def reset_zoom(self) -> None:
        """Reset zoom to show full images."""
        self.zoom_region = None
        self.display_images()

    @staticmethod
    def resize_image(img: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions while maintaining aspect ratio.
        
        Args:
            img: Input image array
            
        Returns:
            Resized image array with padding if necessary
        """
        h, w = img.shape[:2]
        # Calculate scaling factor to fit within target size
        scale = min(ImageViewer.TARGET_HEIGHT/h, ImageViewer.TARGET_WIDTH/w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create black canvas of target size
        if len(img.shape) == 3:
            canvas = np.zeros((ImageViewer.TARGET_HEIGHT, ImageViewer.TARGET_WIDTH, 3), dtype=img.dtype)
        else:
            canvas = np.zeros((ImageViewer.TARGET_HEIGHT, ImageViewer.TARGET_WIDTH), dtype=img.dtype)
            
        # Calculate position to paste resized image
        y_offset = (ImageViewer.TARGET_HEIGHT - new_h) // 2
        x_offset = (ImageViewer.TARGET_WIDTH - new_w) // 2
        
        # Paste resized image onto canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized  # Fixed x2_1 to x_offset
        return canvas

    def display_images(self) -> None:
        """Display current images with optional zoom regions."""
        # Check if we have images to display
        if not self.images1_list or not self.images2_list:
            # Clear all axes
            for ax in self.ax:
                ax.clear()
                ax.set_visible(False)
            self.canvas.draw()
            return

        # Clear all axes
        for ax in self.ax:
            ax.clear()

        # Read new images
        img1 = cv2.imread(self.images1_list[self.index1], cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(self.images2_list[self.index2], cv2.IMREAD_UNCHANGED)

        if img1 is None or img2 is None:
            QMessageBox.warning(self, 'Warning', 'Failed to load images!')
            return

        def process_image(img):
            # Convert color image to grayscale if necessary
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Normalize 32-bit images for display
            if img.dtype == np.float32:
                img = img.astype(np.float32)
                img_min = np.min(img)
                img_max = np.max(img)
                if img_max != img_min:
                    img = (img - img_min) / (img_max - img_min)
            
            return self.resize_image(img)

        img1 = process_image(img1)
        img2 = process_image(img2)
        
        # Apply thresholding if enabled
        if self.enable_threshold.isChecked():
            try:
                threshold_val = float(self.threshold_value.text() or 0)
                if 0 <= threshold_val <= 1:
                    thresh_img = process_image(user_threshold(img1, threshold_val))
                else:
                    QMessageBox.warning(self, 'Warning', 'Threshold value must be between 0 and 1')
                    thresh_img = np.zeros_like(img1)
            except ValueError:
                thresh_img = np.zeros_like(img1)
        else:
            thresh_img = np.zeros_like(img1)

        # Create blank image of the same size and type
        blank_img = np.zeros_like(img1)

        if self.zoom_region:
            # Create deep copies for marking
            img1_marked = img1.copy()
            img2_marked = img2.copy()
            blank_marked = blank_img.copy()
            
            # Get zoom coordinates and calculate bounds
            y, x = self.zoom_region.y, self.zoom_region.x
            h, w = self.zoom_region.height, self.zoom_region.width
            
            # Calculate bounds for each image
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]
            
            # Calculate bounds
            y1 = max(0, min(y, height1 - 1))
            x1 = max(0, min(x, width1 - 1))
            y2_1 = max(0, min(y + h, height1))
            x2_1 = max(0, min(x + w, width1))
            
            y2 = max(0, min(y, height2 - 1))
            x2 = max(0, min(x, width2 - 1))
            y2_2 = max(0, min(y + h, height2))
            x2_2 = max(0, min(x + w, width2))
            
            # Draw rectangles (using white for grayscale images)
            cv2.rectangle(img1_marked, (x1, y1), (x2_1, y2_1), 1.0, 2)  # Use 1.0 for max brightness
            cv2.rectangle(img2_marked, (x2, y2), (x2_2, y2_2), 1.0, 2)
            
            # Extract zoomed regions
            img1_zoomed = img1[y1:y2_1, x1:x2_1].copy()
            img2_zoomed = img2[y2:y2_2, x2:x2_2].copy()
            blank_zoomed = blank_img[y1:y2_1, x1:x2_1].copy()
            
            # Display images with grayscale colormap
            self.ax[0].imshow(img1_marked, cmap='gray')
            self.ax[1].imshow(img2_marked, cmap='gray')
            
            # Update third column with thresholded image
            if self.enable_threshold.isChecked():
                self.ax[2].set_visible(True)
                self.ax[2].imshow(thresh_img, cmap='gray')
                # For zoomed region
                thresh_zoomed = thresh_img[y1:y2_1, x1:x2_1].copy()
                self.ax[5].set_visible(True)
                self.ax[5].imshow(thresh_zoomed, cmap='gray')
            else:
                self.ax[2].set_visible(False)
                self.ax[5].set_visible(False)
            
            # Bottom row - always show zoomed regions
            self.ax[3].set_visible(True)
            self.ax[4].set_visible(True)
            self.ax[3].imshow(img1_zoomed, cmap='gray')
            self.ax[4].imshow(img2_zoomed, cmap='gray')

        else:
            # Display original images
            self.ax[0].imshow(img1, cmap='gray')
            self.ax[1].imshow(img2, cmap='gray')
            
            if self.enable_threshold.isChecked():
                self.ax[2].set_visible(True)
                self.ax[2].imshow(thresh_img, cmap='gray')
            else:
                self.ax[2].set_visible(False)
                
            # Hide bottom row
            for i in range(3, 6):
                self.ax[i].set_visible(False)

        # Set titles
        base1 = os.path.basename(self.images1_list[self.index1])
        base2 = os.path.basename(self.images2_list[self.index2])
        self.ax[0].set_title(f"Folder 1: {base1}")
        self.ax[1].set_title(f"Folder 2: {base2}")
        
        if self.enable_threshold.isChecked():
            self.ax[2].set_title("Result (Empty)")
            if self.zoom_region:
                self.ax[3].set_title("Folder 1 (Zoomed)")
                self.ax[4].set_title("Folder 2 (Zoomed)")
                self.ax[5].set_title("Result (Zoomed)")
        elif self.zoom_region:
            self.ax[2].set_title("Folder 1 (Zoomed)")
            self.ax[3].set_title("Folder 2 (Zoomed)")

        # Turn off axes
        for ax in self.ax:
            if ax.get_visible():
                ax.axis('off')

        # Simple auto layout
        self.canvas.draw()

    def on_threshold_changed(self, state):
        """Handle threshold checkbox state changes."""
        # Instead of recreating figure, just update visibility
        if not state:
            # Hide third column
            self.ax[2].set_visible(False)
            self.ax[5].set_visible(False)
        # Refresh display
        if self.images1_list and self.images2_list:
            self.display_images()

    def on_threshold_value_changed(self, value: str) -> None:
        """Handle threshold value changes."""
        if not self.images1_list or not self.images2_list:
            return
            
        try:
            threshold_val = float(value)
            if 0 <= threshold_val <= 1:
                # Only update if threshold is enabled and images are loaded
                if self.enable_threshold.isChecked():
                    self.display_images()
        except ValueError:
            # Ignore invalid inputs
            pass

    def on_threshold_enabled(self, state: int) -> None:
        """Handle threshold checkbox state changes."""
        # Enable/disable threshold value input based on checkbox state
        self.threshold_value.setEnabled(bool(state))
        
        if not state:
            # Clear threshold value when disabling
            self.threshold_value.clear()
            # Hide third column
            self.ax[2].set_visible(False)
            self.ax[5].set_visible(False)
        
        # Refresh display if images are loaded
        if self.images1_list and self.images2_list:
            self.display_images()

    def on_zoom_enabled(self, state: int) -> None:
        """Handle zoom checkbox state changes."""
        # Enable/disable zoom inputs based on checkbox state
        enabled = bool(state)
        self.zoom_y.setEnabled(enabled)
        self.zoom_x.setEnabled(enabled)
        self.zoom_width.setEnabled(enabled)
        self.zoom_height.setEnabled(enabled)
        
        if not enabled:
            # Clear zoom values when disabled
            self.zoom_y.clear()
            self.zoom_x.clear()
            self.zoom_width.clear()
            self.zoom_height.clear()
            self.zoom_region = None
            
        # Refresh display if images are loaded
        if self.images1_list and self.images2_list:
            self.display_images()

    def on_zoom_values_changed(self) -> None:
        """Handle zoom value changes."""
        if not self.images1_list or not self.images2_list:
            return
            
        try:
            y = int(self.zoom_y.text() or 0)
            x = int(self.zoom_x.text() or 0)
            width = int(self.zoom_width.text() or 0)
            height = int(self.zoom_height.text() or 0)
            
            if width > 0 and height > 0:
                self.zoom_region = ZoomRegion(y, x, width, height)
                if self.enable_zoom.isChecked():
                    self.display_images()
        except ValueError:
            # Ignore invalid inputs
            pass

    def scroll_images(self, value: int) -> None:
        """Update displayed images based on scroll position.

        Args:
            value: New index position from scroll bar
        """
        self.index1 = value
        self.index2 = value
        self.display_images()

    def on_scroll(self, event) -> None:
        """Handle mouse wheel scroll events.

        Args:
            event: Matplotlib scroll event
        """
        # print(f"Scroll event detected: {event.button}")  # Debugging line to check if scroll is working
        if event.button == 'up':
            new_value = max(0, self.scrollbar.value() - 1)
        elif event.button == 'down':
            new_value = min(self.scrollbar.maximum(), self.scrollbar.value() + 1)
        else:
            return

        self.scrollbar.setValue(new_value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
