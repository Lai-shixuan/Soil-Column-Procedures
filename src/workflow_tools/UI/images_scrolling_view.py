import sys
import matplotlib

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from PyQt5.QtWidgets import QApplication
from src.workflow_tools.UI.image_viewers.image_viewer import ImageViewer

matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
