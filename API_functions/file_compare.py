import os
import cv2
import matplotlib.pyplot as plt

class ZoomRegion:
    """
    A class to represent a zoom region in an image. It has two kinds of points: (y, x) for numpy and (x, y) for cv2.
    """
    def __init__(self, y, x, width, height):
        self.y = y
        self.x = x
        self.height = height
        self.width = width

    def get_numpy_points(self):
        """ Return the top-left and bottom-right coordinates in (y, x) format for use with numpy. """
        return (self.y, self.x), (self.y + self.height, self.x + self.width)

    def get_cv_points(self):
        """ Return the top-left and bottom-right coordinates in (x, y) format for use with cv2. """
        return (self.x, self.y), (self.x + self.width, self.y + self.height)

def zoom_in(image, zoom):
    return image[zoom.y:zoom.y + zoom.height, zoom.x:zoom.x + zoom.width]

class ImageProcessor:
    def __init__(self, image_path, label_image=None):
        self.image_path = image_path
        self.label_image_path = label_image
        self._image = None
        self._label_image = None
        self.results = {}

    def image(self):
        # Lazy loading of the main image
        if self._image is None:
            self._image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        return self._image

    def label_image(self):
        # Lazy loading of the label image
        if self._label_image is None and self.label_image_path is not None:
            self._label_image = cv2.imread(self.label_image_path, cv2.IMREAD_UNCHANGED)
        return self._label_image

    def add_result(self, key, image):
        self.results[key] = image

    def load_image(self):
        """ Load the main image if not already loaded. """
        if self._image is None:
            self._image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        return self._image

    def load_label_image(self):
        """ Load the label image if not already loaded and if it exists. """
        if self._label_image is None and self.label_image_path:
            self._label_image = cv2.imread(self.label_image_path, cv2.IMREAD_UNCHANGED)
        return self._label_image

    def show_images(self, *keys, zoom_region=None):
        """
        Display multiple images stored in the results dictionary with optional zoomed regions.
        Show 2 images for each key if zoom_region is specified
        """

        # Load images if they have not been loaded yet
        if 'original' not in self.results:
            self.results['original'] = self.load_image()
        if 'label' not in self.results and self.label_image_path:
            self.results['label'] = self.load_label_image()

        total_images = len(keys) * (2 if zoom_region else 1)
        fig, axes = plt.subplots(1, total_images, figsize=(5 * total_images, 5))
        if total_images == 1:
            axes = [axes]  # Make it iterable

        ax_index = 0
        for key in keys:
            img = self.results.get(key)
            if img is None:
                continue  # Skip if the result does not exist

            # Show the full image with the zoom area marked
            marked_image = img.copy()
            cvpoints1, cvpoints2 = zoom_region.get_cv_points()
            cv2.rectangle(marked_image, cvpoints1, cvpoints2, (0, 255, 0), 3)
            axes[ax_index].imshow(marked_image, cmap='gray')
            axes[ax_index].set_title(f'{key} full')
            axes[ax_index].axis('on')
            ax_index += 1

            # Show the zoomed image if zoom_region is specified
            if zoom_region:
                zoomed_img = zoom_in(img, zoom_region)
                axes[ax_index].imshow(cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB) if zoomed_img.ndim == 3 else zoomed_img, cmap='gray')
                axes[ax_index].set_title(f'{key} zoomed')
                axes[ax_index].axis('on')
                ax_index += 1
        
        plt.tight_layout()
        plt.show(block=True)

class ImageDatabase:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.images_folder = os.path.join(root_folder, 'images')
        self.labels_folder = os.path.join(root_folder, 'labels')
        self.images = {}
        self.load_image_paths()

    def load_image_paths(self):
        for image_name in os.listdir(self.images_folder):
            image_path = os.path.join(self.images_folder, image_name)
            label_path = os.path.join(self.labels_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            label_path = label_path if os.path.exists(label_path) else None
            self.images[image_name] = ImageProcessor(image_path=image_path, label_image=label_path)

    def get_image_processor(self, image_name):
        return self.images.get(image_name)

if __name__ == '__main__':
    path = 'e:/3.Experimental_Data/DL_Data_raw/'
    db = ImageDatabase(path)
    image_processor = db.get_image_processor('002_ou_DongYing_12635.png')
    zoom = ZoomRegion(350, 450, 100, 200)
    image_processor.show_images('original', 'label', zoom_region=zoom)
