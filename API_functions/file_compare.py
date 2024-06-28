import os
import cv2
import matplotlib.pyplot as plt

from .Soils import threshold_position_independent as tpi

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

class ImageResults:
    def __init__(self, image_type:str, image_path:str):
        self.results = {}
        self.add_type(key=image_type, path=image_path)

    def add_type(self, key, path):
        self.results[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.results[key+'path'] = path

    def show_images(self, *keys, zoom_region=None):
        """
        Display multiple images stored in the results dictionary with optional zoomed regions.
        Show 2 images for each key if zoom_region is specified
        """

        total_images = len(keys) * (2 if zoom_region else 1)
        fig, axes = plt.subplots(1, total_images, figsize=(5 * total_images, 5))
        if total_images == 1:
            axes = [axes]  # Make it iterable

        ax_index = 0
        for key in keys:

            # These 2 images are definetely available
            if key == 'label':
                img = self.label_image
            elif key == 'original':
                img = self.image

            # The rest of the images are manually added by `add_result` function
            # and they should use `.get` method
            else:
                img = self.results.get(key)

            if img is None:
                continue  # Skip if the result does not exist

            # Show the full image with the zoom area marked
            marked_image = img.copy()
            if zoom_region:
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
    """
    The differece between this class and Column is that this class's files are all have the same name.
    """

    def __init__(self):
        self.images = {}

    def add_additional_folder(self, additional_folder, name):
        setattr(self, name, additional_folder)
        
        for image_name in os.listdir(additional_folder):

            additional_image_path = os.path.join(additional_folder, image_name)
            # check whether the file is an image
            if cv2.imread(additional_image_path, cv2.IMREAD_UNCHANGED) is None:
                continue
            
            if image_name.endswith('_0000.png'):
                image_name = image_name[:-9] + '.png'

            if image_name in self.images:
                self.images[image_name].add_type(name, additional_image_path)
            else:
                self.images[image_name] = ImageResults(image_type=name, image_path=additional_image_path)

    def get_image_processor(self, image_name):
        if image_name not in self.images:
            raise ValueError(f'Image {image_name} not found in the database')
        return self.images.get(image_name)

if __name__ == '__main__':
    db = ImageDatabase()
    # image_processor.add_result('pre_processed', tpi.user_threshold(image_processor.image, 160))
    zoom = ZoomRegion(350, 450, 100, 200)
    db.add_additional_folder('//wsl.localhost/Ubuntu/home/shixuan/DL_Algorithms/nnunet/nnUNet_results/Dataset001_240531/inference/', 'test_inference')
    db.add_additional_folder('//wsl.localhost/Ubuntu/home/shixuan/DL_Algorithms/nnunet/nnUNet_raw/Dataset001_240531/imagesTs/', 'test_set')
    image_processor = db.get_image_processor('002_ou_DongYing_13636.png')
    image_processor.show_images('test_set', 'test_inference', zoom_region=zoom)
