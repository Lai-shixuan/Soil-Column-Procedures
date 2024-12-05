from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtCore import QTimer

class StatusManager:
    def __init__(self, parent):
        self.parent = parent
        self.status_bar = parent.statusBar()  # Use existing status bar
        self.timer = QTimer()
        self.timer.timeout.connect(self.clear_status)

    def show_status(self, message: str, timeout: int = 5000, error: bool = False):
        """Show message in status bar for specified timeout (ms)."""
        self.status_bar.setStyleSheet("QStatusBar{color: " + ("red" if error else "black") + ";}")
        self.status_bar.showMessage(message)
        self.timer.start(timeout)

    def clear_status(self):
        """Clear the status bar message."""
        self.status_bar.clearMessage()
        self.timer.stop()
        self.status_bar.setStyleSheet("")