import sys
import cv2
import os
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                           QPushButton, QFileDialog, QHBoxLayout, QMessageBox,
                           QScrollBar, QStyle, QStyleFactory, QLabel)
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use('Qt5Agg')

# Enable High DPI scaling
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")
from src.API_functions.Images.file_batch import get_image_names

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set modern style
        self.setStyle(QStyleFactory.create('Fusion'))
        
        # Set light palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        # Set modern font
        self.setFont(QFont('Segoe UI', 9))
        
        self.folder1 = None
        self.folder2 = None
        self.images1_list = []
        self.images2_list = []
        
        # Index of the current image
        self.index1 = 0
        self.index2 = 0
        
        # Create the Qt5 main window
        self.setWindowTitle('Image Viewer')
        
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)

        # Set up the layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Create buttons
        self.btn_folder1 = QPushButton('Select Folder 1')
        self.btn_folder2 = QPushButton('Select Folder 2')
        self.btn_start = QPushButton('Start Viewing')
        self.btn_start.setEnabled(False)

        # Connect buttons to functions
        self.btn_folder1.clicked.connect(lambda: self.select_folder(1))
        self.btn_folder2.clicked.connect(lambda: self.select_folder(2))
        self.btn_start.clicked.connect(self.start_viewing)

        # Add buttons to button layout
        button_layout.addWidget(self.btn_folder1)
        button_layout.addWidget(self.btn_folder2)
        button_layout.addWidget(self.btn_start)

        # Create labels for folder paths
        self.label_folder1 = QLabel('No folder selected')
        self.label_folder2 = QLabel('No folder selected')
        
        # Set label styles
        label_style = """
            QLabel {
                color: #333333;
                background-color: #FFFFFF;
                padding: 5px;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                min-height: 20px;
            }
        """
        self.label_folder1.setStyleSheet(label_style)
        self.label_folder2.setStyleSheet(label_style)
        
        # Create layout for path labels
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.label_folder1)
        path_layout.addWidget(self.label_folder2)

        # Create vertical layout for scroll bar and canvas
        canvas_layout = QHBoxLayout()
        
        # Create and setup scroll bar
        self.scrollbar = QScrollBar()
        self.scrollbar.setOrientation(Qt.Vertical)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(0)  # Will be updated when images are loaded
        self.scrollbar.valueChanged.connect(self.scroll_images)
        
        # Add canvas and scroll bar to layout
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.scrollbar)
        
        # Modify main layout to use the new canvas layout
        main_layout.addLayout(button_layout)
        main_layout.addLayout(path_layout)  # Add path labels below buttons
        main_layout.addLayout(canvas_layout)

        # Create a widget and set the layout
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        self.pattern_length = None  # Store the common prefix length between folders

        # Modify button styles for light theme
        button_style = """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                color: #333333;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #E6E6E6;
                border: 1px solid #ADADAD;
            }
            QPushButton:pressed {
                background-color: #D4D4D4;
                border: 1px solid #8C8C8C;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                border: 1px solid #E3E3E3;
                color: #9E9E9E;
            }
        """
        self.btn_folder1.setStyleSheet(button_style)
        self.btn_folder2.setStyleSheet(button_style)
        self.btn_start.setStyleSheet(button_style)
        
        # Modify scrollbar style for light theme
        scrollbar_style = """
            QScrollBar:vertical {
                border: none;
                background: #F0F0F0;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #CDCDCD;
                min-height: 20px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background: #A6A6A6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
        self.scrollbar.setStyleSheet(scrollbar_style)
        
        # Set window size and style
        self.setMinimumSize(800, 600)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | 
                          Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

    def get_image_files(self, folder):
        # Simplify to only get tif files
        files = get_image_names(folder, None, 'tif')
        return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])

    def get_common_prefix_length(self, str1, str2):
        """Find the length of common prefix between two strings"""
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return i
        return min_len

    def analyze_filename_pattern(self, files1, files2):
        """Analyze first 5 pairs of files to determine pattern length"""
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

    def validate_folders(self):
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
        return True

    def select_folder(self, folder_num):
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
            
            # Enable start button if both folders are selected and validated
            if self.folder1 and self.folder2:
                if self.validate_folders():
                    self.btn_start.setEnabled(True)
                else:
                    self.btn_start.setEnabled(False)

    def start_viewing(self):
        if self.folder1 and self.folder2 and self.validate_folders():
            self.scrollbar.setValue(0)  # Reset to first image
            self.display_images()
            self.canvas.mpl_connect('scroll_event', self.on_scroll)
        else:
            QMessageBox.warning(self, 'Warning', 'Please ensure both folders contain the same image files.')

    def display_images(self):
        # Clear the previous images to prevent overlap
        for axis in self.ax:
            axis.clear()

        # Read new images
        img1 = cv2.imread(self.images1_list[self.index1], cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(self.images2_list[self.index2], cv2.IMREAD_UNCHANGED)

        # Display the images
        self.ax[0].imshow(img1, cmap='gray')
        self.ax[0].set_title(f"Folder 1: {os.path.basename(self.images1_list[self.index1])}")
        self.ax[0].axis('off')

        self.ax[1].imshow(img2, cmap='gray')
        self.ax[1].set_title(f"Folder 2: {os.path.basename(self.images2_list[self.index2])}")
        self.ax[1].axis('off')

        # Refresh the canvas to update the display
        self.canvas.draw()
        
        # Update matplotlib style for light theme
        plt.style.use('default')
        
        # Improve figure appearance for light theme
        self.fig.tight_layout(pad=3.0)
        self.fig.patch.set_facecolor('#FFFFFF')
        
        for ax in self.ax:
            ax.set_facecolor('#FFFFFF')
            ax.title.set_color('black')

    def scroll_images(self, value):
        self.index1 = value
        self.index2 = value
        self.display_images()

    def on_scroll(self, event):
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
