import sys
import cv2
import os
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use('Qt5Agg')

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")
from API_functions import file_batch as fb

class ImageViewer(QMainWindow):
    def __init__(self, folder1, folder2):
        super().__init__()

        self.images1_list = fb.get_image_names(folder1, None, 'tif')
        self.images2_list = fb.get_image_names(folder2, None, 'tif')
        
        # Index of the current image
        self.index1 = 0
        self.index2 = 0
        
        # Create the Qt5 main window
        self.setWindowTitle('Image Viewer')
        
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)

        # Set up the layout for the window
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        # Create a widget and set the layout
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Display the images
        self.display_images()

        # Connect the scroll event
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

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

    def on_scroll(self, event):
        # print(f"Scroll event detected: {event.button}")  # Debugging line to check if scroll is working
        if event.button == 'up':
            self.index1 = (self.index1 - 1) % len(self.images1_list)
            self.index2 = (self.index2 - 1) % len(self.images2_list)
        elif event.button == 'down':
            self.index1 = (self.index1 + 1) % len(self.images1_list)
            self.index2 = (self.index2 + 1) % len(self.images2_list)

        # After updating the indices, update the display
        self.display_images()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    folder1 = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/3.Precheck/images/'
    folder2 = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/3.Precheck/labels/'

    viewer = ImageViewer(folder1, folder2)
    viewer.show()

    sys.exit(app.exec_())  # Start the Qt event loop
