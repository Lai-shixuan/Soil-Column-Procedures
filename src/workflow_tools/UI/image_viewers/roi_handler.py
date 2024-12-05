
import matplotlib.pyplot as plt

class ROIHandler:
    def __init__(self, parent):
        self.parent = parent
        self.drawing_roi = False
        self.start_point = None
        self.current_point = None
        self.temp_artists = []

    def start_drawing(self):
        """Start ROI drawing mode."""
        if not self.drawing_roi:
            self.drawing_roi = True
            self.parent.controls.draw_roi.setText('Drawing...')
            self.parent.status_manager.show_status('Click and drag to draw ROI')
            self._clear_temp_artists()

    def on_mouse_press(self, event):
        """Handle mouse press event for ROI drawing."""
        if not self.drawing_roi or event.inaxes not in self.parent.ax[:3]:
            return
            
        self.start_point = (event.xdata, event.ydata)

    def on_mouse_move(self, event):
        """Handle mouse move event for ROI drawing."""
        if not self.drawing_roi or not self.start_point or not event.inaxes:
            return
            
        self.current_point = (event.xdata, event.ydata)
        # Clear previous temporary rectangles
        for artist in self.temp_artists:
            artist.remove()
        self.temp_artists.clear()
        
        # Draw temporary rectangle on all visible images in top row
        for ax in self.parent.ax.flatten()[:3]:  # Use flatten() to get the individual Axes objects
            if ax.get_visible():
                x = [self.start_point[0], self.current_point[0]]
                y = [self.start_point[1], self.current_point[1]]
                width = abs(x[1] - x[0])
                height = abs(y[1] - y[0])
                rect = plt.Rectangle(
                    (min(x), min(y)), width, height,
                    fill=False, color='red', linestyle='--', linewidth=1
                )
                ax.add_patch(rect)
                self.temp_artists.append(rect)
        
        self.parent.canvas.draw()

    def on_mouse_release(self, event):
        """Handle mouse release event for ROI drawing."""
        if not self.drawing_roi or not self.start_point or not event.inaxes:
            return
            
        self.drawing_roi = False
        self.parent.controls.draw_roi.setText('Draw ROI')
        
        # Clear temporary rectangles
        for artist in self.temp_artists:
            artist.remove()
        self.temp_artists.clear()
        
        # Calculate ROI parameters
        x1, y1 = self.start_point
        x2, y2 = event.xdata, event.ydata
        
        # Convert to image coordinates
        x = min(int(x1), int(x2))
        y = min(int(y1), int(y2))
        width = abs(int(x2 - x1))
        height = abs(int(y2 - y1))
        
        # Update zoom controls
        self.parent.controls.enable_zoom.setChecked(True)
        self.parent.controls.zoom_x.setText(str(x))
        self.parent.controls.zoom_y.setText(str(y))
        self.parent.controls.zoom_width.setText(str(width))
        self.parent.controls.zoom_height.setText(str(height))
        
        self.start_point = None
        self.current_point = None
        self.parent.display_images()

    def _clear_temp_artists(self):
        """Clear temporary drawing artifacts."""
        for artist in self.temp_artists:
            artist.remove()
        self.temp_artists.clear()
        self.parent.canvas.draw()