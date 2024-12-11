import os
import matplotlib.pyplot as plt
import cv2

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QScrollBar
from PyQt5.QtCore import Qt

from ..controls import ImageControls
from ..image_display import ImageDisplay
from src.API_functions.Visual.file_compare import ZoomRegion
from .file_manager import FileManager
from .roi_handler import ROIHandler
from .status_manager import StatusManager

class ImageViewer(QMainWindow):
    """Main window class for the image comparison viewer application.
    
    This viewer allows users to:
        - Select two folders containing images
        - Navigate through corresponding images
        - Zoom into specific regions of the images
        - Compare images side by side
        - Apply thresholding to images
    
    Attributes:
        TARGET_WIDTH (int): Fixed width for display (512)
        TARGET_HEIGHT (int): Fixed height for display (512)
        WINDOW_WIDTH (int): Main window width (1600)
        WINDOW_HEIGHT (int): Main window height (900)
        folder1 (str): Path to first folder
        folder2 (str): Path to second folder
        images1_list (List[str]): List of images from folder1
        images2_list (List[str]): List of images from folder2
        index1 (int): Current index in folder1
        index2 (int): Current index in folder2
        pattern_length (int): Common prefix length between paired files
    """
    TARGET_WIDTH = 512
    TARGET_HEIGHT = 512
    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 900

    def __init__(self):
        super().__init__()
        self._setup_window()
        self._init_components()
        self._create_layouts()
        self._connect_signals()
        

    def _setup_window(self):
        """Setup main window properties."""
        self.setWindowTitle('Image Viewer')
        self.setMinimumSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

    def _init_components(self):
        """Initialize all component managers."""
        self.file_manager = FileManager(self)
        self.roi_handler = ROIHandler(self)
        self.status_manager = StatusManager(self)
        self.index1 = 0
        self.index2 = 0

    def _create_layouts(self):
        """Create and setup layouts."""
        # Create the layouts first
        main_layout = QHBoxLayout()
        self.controls = ImageControls()
        
        # Setup matplotlib after the layouts
        self.fig, self.ax = plt.subplots(2, 3, figsize=(20, 9))
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                                wspace=0.2, hspace=0.2)
        self.canvas = FigureCanvas(self.fig)
        self.display = ImageDisplay(self.ax.flatten(), self.TARGET_WIDTH, self.TARGET_HEIGHT)
        
        # Add layouts
        main_layout.addLayout(self.controls.layout, stretch=1)
        main_layout.addWidget(self.canvas, stretch=8)
        
        # Setup scrollbar
        self.scrollbar = QScrollBar(Qt.Vertical)
        main_layout.addWidget(self.scrollbar, stretch=1)
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        """Connect all signal handlers."""
        # Folder selection
        self.controls.btn_folder1.clicked.connect(lambda: self.file_manager.select_folder(1))
        self.controls.btn_folder2.clicked.connect(lambda: self.file_manager.select_folder(2))
        
        # Zoom controls
        self.controls.enable_zoom.stateChanged.connect(self.on_zoom_enabled)
        for widget in [self.controls.zoom_x, self.controls.zoom_y,
                      self.controls.zoom_width, self.controls.zoom_height]:
            widget.textChanged.connect(self.on_zoom_values_changed)
        
        # Threshold controls
        self.controls.enable_threshold.stateChanged.connect(self.on_threshold_enabled)
        self.controls.threshold_value.textChanged.connect(self.on_threshold_value_changed)
        self.controls.save_threshold.clicked.connect(self.save_threshold_result)
        self.controls.threshold_up.clicked.connect(lambda: self.adjust_threshold(0.001))
        self.controls.threshold_down.clicked.connect(lambda: self.adjust_threshold(-0.001))
        
        # ROI controls
        self.controls.draw_roi.clicked.connect(self.roi_handler.start_drawing)
        self.canvas.mpl_connect('button_press_event', self.roi_handler.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.roi_handler.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.roi_handler.on_mouse_release)
        
        # Scrollbar and mouse wheel
        self.scrollbar.valueChanged.connect(self.scroll_images)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Add folder2 enable handler
        self.controls.enable_folder2.stateChanged.connect(self.on_folder2_enabled)

        # Mean filter controls
        self.controls.enable_mean_filter.clicked.connect(self.on_mean_filter_clicked)
        self.controls.filter_size.textChanged.connect(self.on_filter_size_changed)

        # Histogram equalization control
        self.controls.enable_histogram_eq.clicked.connect(self.on_histogram_eq_clicked)

    def scroll_images(self, value: int) -> None:
        """Update displayed images based on scroll position."""
        self.index1 = value
        self.index2 = value
        self.display_images()

    def on_scroll(self, event) -> None:
        """Handle mouse wheel scroll events."""
        if event.button == 'up':
            new_value = max(0, self.scrollbar.value() - 1)
        elif event.button == 'down':
            new_value = min(self.scrollbar.maximum(), self.scrollbar.value() + 1)
        else:
            return
        self.scrollbar.setValue(new_value)

    def enable_controls(self, enabled: bool) -> None:
        """Enable or disable all controls that require folders to be selected."""
        self.controls.enable_zoom.setEnabled(enabled)
        zoom_enabled = enabled and self.controls.enable_zoom.isChecked()
        self.controls.zoom_y.setEnabled(zoom_enabled)
        self.controls.zoom_x.setEnabled(zoom_enabled)
        self.controls.zoom_width.setEnabled(zoom_enabled)
        self.controls.zoom_height.setEnabled(zoom_enabled)
        
        self.controls.enable_threshold.setEnabled(enabled)
        self.controls.threshold_value.setEnabled(enabled and self.controls.enable_threshold.isChecked())
        self.update_save_threshold_button()
        self.controls.draw_roi.setEnabled(enabled)

    def display_images(self) -> None:
        """Display current images using ImageDisplay."""
        if not self.file_manager.images1_list:
            return

        if not (0 <= self.index1 < len(self.file_manager.images1_list)):
            print("Invalid image indices")
            return

        # Check if second column is enabled and has valid images
        show_second_column = (self.controls.enable_folder2.isChecked() and 
                             self.file_manager.images2_list and 
                             0 <= self.index2 < len(self.file_manager.images2_list))

        zoom_region = None
        if self.controls.enable_zoom.isChecked():
            try:
                y = int(self.controls.zoom_y.text() or 0)
                x = int(self.controls.zoom_x.text() or 0)
                width = int(self.controls.zoom_width.text() or 0)
                height = int(self.controls.zoom_height.text() or 0)
                if width > 0 and height > 0:
                    zoom_region = ZoomRegion(y, x, width, height)
            except ValueError:
                pass

        threshold_enabled = self.controls.enable_threshold.isChecked()
        threshold_value = None
        if threshold_enabled:
            try:
                threshold_value = float(self.controls.threshold_value.text() or 0)
                if not 0 <= threshold_value <= 1:
                    threshold_value = None
            except ValueError:
                pass

        img2_path = (self.file_manager.images2_list[self.index2] 
                    if show_second_column else None)

        self.display.display_pair(
            self.file_manager.images1_list[self.index1],
            img2_path,
            zoom_region=zoom_region,
            threshold_enabled=threshold_enabled,
            threshold_value=threshold_value
        )
        self.canvas.draw()

    # Add handlers for Zoom and Threshold
    def on_zoom_enabled(self, state: int) -> None:
        """Handle zoom checkbox state changes."""
        enabled = bool(state)
        for widget in [self.controls.zoom_y, self.controls.zoom_x,
                      self.controls.zoom_width, self.controls.zoom_height]:
            widget.setEnabled(enabled)
        
        if not enabled:
            for widget in [self.controls.zoom_y, self.controls.zoom_x,
                         self.controls.zoom_width, self.controls.zoom_height]:
                widget.clear()
        self.display_images()

    def on_zoom_values_changed(self) -> None:
        """Handle zoom value changes."""
        if not self.file_manager.images1_list or not self.file_manager.images2_list:
            return
            
        if self.controls.enable_zoom.isChecked():
            try:
                # Validate zoom values
                y = max(0, int(self.controls.zoom_y.text() or 0))
                x = max(0, int(self.controls.zoom_x.text() or 0))
                width = max(1, int(self.controls.zoom_width.text() or 0))
                height = max(1, int(self.controls.zoom_height.text() or 0))
                
                # Update only if values are reasonable
                if width <= self.TARGET_WIDTH * 2 and height <= self.TARGET_HEIGHT * 2:
                    self.display_images()
            except ValueError:
                pass

    def on_threshold_enabled(self, state: bool) -> None:
        """Handle threshold checkbox state changes."""
        enabled = bool(state)
        self.controls.threshold_value.setEnabled(enabled)
        self.controls.threshold_up.setEnabled(enabled)
        self.controls.threshold_down.setEnabled(enabled)
        self.controls.enable_mean_filter.setEnabled(enabled)
        self.controls.filter_size.setEnabled(enabled)
        self.controls.enable_histogram_eq.setEnabled(enabled)
        if not enabled:
            self.controls.threshold_value.clear()
        self.update_save_threshold_button()
        self.display_images()

    def on_threshold_value_changed(self, value: str) -> None:
        """Handle threshold value changes."""
        if not self.file_manager.images1_list or not self.file_manager.images2_list:
            return
            
        if self.controls.enable_threshold.isChecked():
            try:
                threshold_val = float(value)
                if 0 <= threshold_val <= 1:
                    self.display_images()
            except ValueError:
                pass
        self.update_save_threshold_button()

    def update_save_threshold_button(self) -> None:
        """Update the state of save threshold button."""
        can_save = (self.controls.enable_threshold.isChecked() and 
                    self.controls.threshold_value.text() and 
                    self.file_manager.folder1 is not None and 
                    self.file_manager.images1_list)
        self.controls.save_threshold.setEnabled(bool(can_save))

    def save_threshold_result(self) -> None:
        """Save current threshold result as float32 image."""
        if not self.file_manager.images1_list or self.index1 >= len(self.file_manager.images1_list):
            return
            
        if self.display.current_threshold_result is None:
            self.status_manager.show_status('Error: No threshold result available to save', error=True)
            return
            
        try:
            # Create threshold directory
            threshold_dir = os.path.join(self.file_manager.folder1, 'threshold')
            os.makedirs(threshold_dir, exist_ok=True)
            
            # Save result directly as float32
            base_name = os.path.basename(self.file_manager.images1_list[self.index1])
            save_path = os.path.join(threshold_dir, base_name)
            cv2.imwrite(save_path, self.display.current_threshold_result)
            
            self.status_manager.show_status(f'Success: Threshold result saved to: {save_path}')
        except Exception as e:
            self.status_manager.show_status(f'Error: Failed to save threshold result: {str(e)}', error=True)

    def adjust_threshold(self, delta: float) -> None:
        """Adjust threshold value by the specified delta."""
        if not self.controls.enable_threshold.isChecked():
            return
            
        try:
            current = float(self.controls.threshold_value.text() or 0)
            new_value = max(0, min(1, current + delta))
            self.controls.threshold_value.setText(f"{new_value:.4f}")  # Changed from .3f to .4f
        except ValueError:
            self.controls.threshold_value.setText("0.5000")  # Changed initial value to 4 decimals

    def on_folder2_enabled(self, state: int) -> None:
        """Handle enabling/disabling of second folder."""
        enabled = bool(state)
        self.controls.btn_folder2.setEnabled(enabled)
        self.controls.label_folder2.setEnabled(enabled)
        if not enabled:
            self.file_manager.folder2 = None
            self.file_manager.images2_list = []
            self.controls.label_folder2.setText('No folder selected')
        self.display_images()

    def on_mean_filter_clicked(self):
        """Toggle mean filter."""
        self.display.mean_filter_enabled = not self.display.mean_filter_enabled
        self.controls.enable_mean_filter.setStyleSheet(
            "background-color: lightblue;" if self.display.mean_filter_enabled else ""
        )
        self.display_images()

    def on_filter_size_changed(self, value: str):
        """Handle filter size changes."""
        try:
            size = int(value)
            if size >= 3 and self.controls.enable_threshold.isChecked():
                self.display.filter_size = size
                self.display_images()
        except ValueError:
            pass

    def on_histogram_eq_clicked(self):
        """Toggle histogram equalization."""
        self.display.histogram_eq_enabled = not self.display.histogram_eq_enabled
        self.controls.enable_histogram_eq.setStyleSheet(
            "background-color: lightblue;" if self.display.histogram_eq_enabled else ""
        )
        self.display_images()