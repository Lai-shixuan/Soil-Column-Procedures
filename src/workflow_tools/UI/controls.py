from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QCheckBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import QSize

class ImageControls:
    """UI controls for the image viewer application.
    
    Manages all control widgets including:
        - Folder selection buttons and labels
        - Zoom controls (enable/disable, coordinates)
        - Threshold controls (enable/disable, value)
    
    Attributes:
        layout (QVBoxLayout): Main layout containing all controls
        btn_folder1 (QPushButton): Button to select first folder
        btn_folder2 (QPushButton): Button to select second folder
        label_folder1 (QLabel): Label showing first folder path
        label_folder2 (QLabel): Label showing second folder path
        enable_zoom (QCheckBox): Checkbox to enable zoom
        zoom_x/y/width/height (QLineEdit): Zoom region input fields
        enable_threshold (QCheckBox): Checkbox to enable threshold
        threshold_value (QLineEdit): Threshold value input field
    """
    def __init__(self):
        self.layout = QVBoxLayout()
        self._setup_folder_controls()
        self._setup_zoom_controls()
        self._setup_threshold_controls()

    def _setup_folder_controls(self):
        """Setup folder selection controls."""
        self.btn_folder1 = QPushButton('Select Folder 1')
        self.btn_folder2 = QPushButton('Select Folder 2')
        self.enable_folder2 = QCheckBox('Enable Second Folder')
        self.label_folder1 = QLabel('No folder selected')
        self.label_folder2 = QLabel('No folder selected')

        folder_layout = QVBoxLayout()
        folder_layout.addWidget(self.btn_folder1)
        folder_layout.addWidget(self.label_folder1)
        
        folder2_group = QHBoxLayout()
        folder2_group.addWidget(self.enable_folder2)
        folder2_group.addWidget(self.btn_folder2)
        folder_layout.addLayout(folder2_group)
        folder_layout.addWidget(self.label_folder2)
        folder_layout.addStretch()

        self.btn_folder2.setEnabled(False)
        self.layout.addLayout(folder_layout)

    def _setup_zoom_controls(self):
        """Setup zoom controls."""
        zoom_group = QVBoxLayout()
        zoom_group.setSpacing(5)
        
        # Create zoom header with checkbox and draw button
        zoom_header = QHBoxLayout()
        self.enable_zoom = QCheckBox('Enable Zoom')
        self.enable_zoom.setEnabled(False)
        self.draw_roi = QPushButton('Draw ROI')
        self.draw_roi.setEnabled(False)
        zoom_header.addWidget(self.enable_zoom)
        zoom_header.addWidget(self.draw_roi)
        zoom_group.addLayout(zoom_header)
        
        # Create grid layout for zoom controls
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
            widget.setValidator(QIntValidator(0, 9999))
            widget.setPlaceholderText('0-9999')
            zoom_inputs.addWidget(widget)
        
        zoom_grid.addLayout(zoom_labels)
        zoom_grid.addLayout(zoom_inputs)
        zoom_group.addLayout(zoom_grid)
        zoom_group.addStretch()

        self.layout.addLayout(zoom_group)

    def _setup_threshold_controls(self):
        """Setup threshold controls."""
        threshold_group = QVBoxLayout()
        
        # Threshold enable and value
        threshold_header = QHBoxLayout()
        self.enable_threshold = QCheckBox('Enable Threshold')
        self.enable_threshold.setEnabled(False)
        
        # Create a container for threshold value and adjustment buttons
        threshold_value_container = QHBoxLayout()
        
        self.threshold_value = QLineEdit()
        self.threshold_value.setFixedWidth(60)
        self.threshold_value.setPlaceholderText('0-1')
        self.threshold_value.setEnabled(False)
        self.threshold_value.setValidator(QDoubleValidator(0.0, 1.0, 4))  # Changed from 3 to 4
        
        # Create adjustment buttons
        self.threshold_up = QPushButton('▲')
        self.threshold_down = QPushButton('▼')
        for btn in [self.threshold_up, self.threshold_down]:
            btn.setFixedSize(QSize(20, 20))
            btn.setEnabled(False)
        
        threshold_value_container.addWidget(self.threshold_value)
        threshold_value_container.addWidget(self.threshold_up)
        threshold_value_container.addWidget(self.threshold_down)
        
        threshold_header.addWidget(self.enable_threshold)
        threshold_header.addLayout(threshold_value_container)
        threshold_group.addLayout(threshold_header)
        
        # Add mean filter controls
        filter_container = QHBoxLayout()
        self.enable_mean_filter = QPushButton('Mean Filter')
        self.enable_mean_filter.setEnabled(False)
        self.filter_size = QLineEdit()
        self.filter_size.setFixedWidth(30)
        self.filter_size.setPlaceholderText('3')
        self.filter_size.setValidator(QIntValidator(3, 15))  # Only odd numbers from 3-15
        self.filter_size.setEnabled(False)
        
        filter_container.addWidget(self.enable_mean_filter)
        filter_container.addWidget(self.filter_size)

        # Add histogram equalization control
        self.enable_histogram_eq = QPushButton('Histogram Equalization')
        self.enable_histogram_eq.setEnabled(False)
        filter_container.addWidget(self.enable_histogram_eq)
        
        threshold_group.addLayout(filter_container)
        
        # Add save button
        self.save_threshold = QPushButton('Save Threshold')
        self.save_threshold.setEnabled(False)
        threshold_group.addWidget(self.save_threshold)
        
        threshold_group.addStretch()
        self.layout.addLayout(threshold_group)

    def get_layout(self):
        """Get the main layout containing all controls."""
        return self.layout

    def clear_all(self):
        """Clear all input fields."""
        self.label_folder1.setText('No folder selected')
        self.label_folder2.setText('No folder selected')
        self.enable_zoom.setChecked(False)
        self.enable_threshold.setChecked(False)
        self.threshold_value.clear()
        for widget in [self.zoom_y, self.zoom_x, self.zoom_width, self.zoom_height]:
            widget.clear()
        self.filter_size.setText('3')  # Set default filter size