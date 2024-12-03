import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QVBoxLayout, QWidget, QFileDialog, QMessageBox, 
                            QSizePolicy, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QScreen

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("文件夹选择器")
        
        # Get screen DPI scaling factor
        screen = QApplication.primaryScreen()
        self.dpi_scale = screen.logicalDotsPerInch() / 96.0
        
        # Scale window size based on DPI
        self.resize(int(800 * self.dpi_scale), int(400 * self.dpi_scale))
        
        # Create main widget and set it as central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        # Scale layout margins and spacing
        layout.setContentsMargins(
            int(40 * self.dpi_scale), 
            int(40 * self.dpi_scale), 
            int(40 * self.dpi_scale), 
            int(40 * self.dpi_scale)
        )
        layout.setSpacing(int(20 * self.dpi_scale))
        main_widget.setLayout(layout)
        
        # Update font sizes with point sizes (pt) instead of pixels
        app_font = QFont("Arial", int(10 * self.dpi_scale))
        self.setFont(app_font)
        
        # Update button styles with scaled dimensions
        button_style = f"""
            QPushButton {{
                background-color: #2196F3;
                border: none;
                color: white;
                padding: {int(15 * self.dpi_scale)}px;
                border-radius: {int(10 * self.dpi_scale)}px;
                font-size: {int(11 * self.dpi_scale)}pt;
            }}
            QPushButton:hover {{
                background-color: #1976D2;
            }}
            QPushButton:disabled {{
                background-color: #BDBDBD;
            }}
        """
        
        self.folder1_button = QPushButton("选择第一个文件夹")
        self.folder2_button = QPushButton("选择第二个文件夹")
        self.process_button = QPushButton("开始处理")
        
        # Apply style to buttons
        for button in [self.folder1_button, self.folder2_button, self.process_button]:
            button.setStyleSheet(button_style)
            button.setMinimumHeight(int(60 * self.dpi_scale))
        
        # Add buttons to layout
        layout.addWidget(self.folder1_button)
        layout.addWidget(self.folder2_button)
        layout.addWidget(self.process_button)
        
        # Add info display area
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setMinimumHeight(int(150 * self.dpi_scale))
        self.info_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #F5F5F5;
                border: 1px solid #BDBDBD;
                border-radius: {int(5 * self.dpi_scale)}px;
                padding: {int(10 * self.dpi_scale)}px;
                font-family: Consolas, Monaco, monospace;
                font-size: {int(9 * self.dpi_scale)}pt;
            }}
        """)
        layout.addWidget(self.info_display)  # Add info display to layout
        
        # Initialize folder paths
        self.folder1_path = ""
        self.folder2_path = ""
        
        # Connect button signals to functions
        self.folder1_button.clicked.connect(lambda: self.select_folder(1))
        self.folder2_button.clicked.connect(lambda: self.select_folder(2))
        self.process_button.clicked.connect(self.process_folders)
        
        # Disable process button until both folders are selected
        self.process_button.setEnabled(False)
        
        # Scale minimum window size
        self.setMinimumSize(int(400 * self.dpi_scale), int(300 * self.dpi_scale))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def select_folder(self, button_num):
        folder_path = QFileDialog.getExistingDirectory(self, f"选择文件夹 {button_num}")
        if folder_path:
            if button_num == 1:
                self.folder1_path = folder_path
                self.folder1_button.setText(f"文件夹1: {folder_path}")
            else:
                self.folder2_path = folder_path
                self.folder2_button.setText(f"文件夹2: {folder_path}")
        
        # 检查是否两个文件夹都已选择
        if self.folder1_path and self.folder2_path:
            self.process_button.setEnabled(True)
    
    def process_folders(self):
        try:
            # 在这里添加你的处理函数
            self.your_processing_function()
            
            # 显示完成消息
            QMessageBox.information(self, "完成", "segment finish")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中出现错误：{str(e)}")
    
    def update_info(self, message):
        """Update the info display with new message"""
        self.info_display.append(message)
        # Ensure the latest message is visible
        self.info_display.verticalScrollBar().setValue(
            self.info_display.verticalScrollBar().maximum()
        )
    
    def your_processing_function(self):
        """Example of how to use the info display"""
        self.info_display.clear()  # Clear previous messages
        self.update_info(f"Starting processing...")
        self.update_info(f"Input folder 1: {self.folder1_path}")
        self.update_info(f"Input folder 2: {self.folder2_path}")
        
        # Example of progress updates
        for i in range(5):
            # Simulate some work
            import time
            time.sleep(1)
            self.update_info(f"Processing step {i+1}/5...")
            QApplication.processEvents()  # Keep UI responsive
        
        self.update_info("Processing complete!")
        


if __name__ == '__main__':
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # Create the application 
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())