import cv2
import numpy as np
import sys

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from tqdm import tqdm
from src.API_functions.Images import file_batch as fb
from src.API_functions.Soils import threshold_position_independent as tpi
from src.API_functions.DL import shape_processor as processor
from src.API_functions.DL import shape_detectors as detector
from src.API_functions.DL import multi_input_adapter as adapter

class ShapeROICutter:
    """Utility class for cutting ROIs from images based on shape detection in label images"""
    
    @staticmethod
    def save_processed_results(processed_image: np.ndarray, output_dir: str, base_name: str, suffix: str = ''):
        """Helper function to save processed images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{base_name}{suffix}.tif"
        cv2.imwrite(str(output_path), processed_image)

    @staticmethod
    def crop_to_roi(image: np.ndarray, params: detector.RectangleParams) -> np.ndarray:
        """Crop the image to the ROI defined by the square parameters"""
        x, y = params.center
        half_size = params.width // 2
        
        # Calculate crop coordinates
        x1 = max(int(x - half_size), 0)
        y1 = max(int(y - half_size), 0)
        x2 = min(int(x + half_size), image.shape[1])
        y2 = min(int(y + half_size), image.shape[0])
        
        return image[y1:y2, x1:x2]

    @classmethod
    def process_folder(cls, 
                        label_folder: str, 
                        image_folder: str, 
                        output_dir: str, 
                        draw_mask: bool = True, 
                        fill_color: int = 0):
        """
        Process all images in a folder to detect ROIs from labels and apply to images.
        
        Args:
            label_folder: Path to folder containing label images
            image_folder: Path to folder containing dataset images
            output_dir: Path to save processed images
            draw_mask: Whether to draw detection visualization
            fill_color: Color to fill outside ROI
        """
        # Load all images
        label_paths = fb.get_image_names(label_folder, None, 'png')
        image_paths = fb.get_image_names(image_folder, None, 'png')
        
        # Read all images, apply threshold to labels
        label_images_files = [adapter.harmonized_normalize(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)) for path in label_paths]
        label_images_files = [tpi.user_threshold(image=img, optimal_threshold=0.5) for img in label_images_files]
        images_files = [adapter.harmonized_normalize(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)) for path in image_paths]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images
        for i in tqdm(range(len(label_images_files)), desc="Processing images"):
            label_image = label_images_files[i]
            image = images_files[i]
            base_name = Path(label_paths[i]).stem
            base_name = base_name.replace('.', '-')
            base_name = base_name.replace('circle', 'cutted')
            base_name = base_name.replace('35-', '35-00')
            
            # Detect shape parameters from label, and cut image
            label_images, params = processor.process_shape_detection(
                label_image, 
                detector.EllipseDetector(), 
                is_label=True, 
                draw_mask=draw_mask
            )
            
            # Calculate maximum interior tangent square
            # For ellipse, the side length of max square is min(long_axis, short_axis)
            square_size = int((2 * params.long_axis * params.short_axis / (params.long_axis ** 2 + params.short_axis ** 2) ** 0.5)/2) - 1
            
            # Create new rectangle parameters centered at the same point
            square_params = detector.RectangleParams(
                center=params.center,
                covered_pixels=0,
                width=square_size,
                height=square_size
            )
            
            # Cut both image and label with the square parameters
            # For compatability, it will not change the size of the image
            image_cut = processor.cut_with_shape(image, square_params, fillcolor=fill_color)
            label_cut = processor.cut_with_shape(label_image, square_params, fillcolor=fill_color)
            
            # Crop the image and label to the ROI size
            image_cropped = cls.crop_to_roi(image_cut, square_params)
            label_cropped = cls.crop_to_roi(label_cut, square_params)

            # Save results
            cls.save_processed_results(label_cropped, output_dir / 'label', base_name)
            cls.save_processed_results(image_cropped, output_dir / 'image', base_name)

            if draw_mask:
                # Draw dataset image with detected parameters
                image = fb.bitconverter.binary_to_grayscale_one_image(image, 'uint8')
                image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image_draw = processor.ShapeDrawer.draw_shape(image_draw, params)
                
                # Save visualization results
                cls.save_processed_results(label_images['draw'], output_dir / 'label_draw', base_name)
                cls.save_processed_results(image_draw, output_dir / 'image_draw', base_name)

if __name__ == "__main__":
    # Example usage
    label_folder = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/labels/'
    image_folder = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/images/'
    output_dir = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/2.ROI/'

    # Special name rules are for online data soil column 35. Need to be adjusted for other datasets!!

    cutter = ShapeROICutter()
    cutter.process_folder(label_folder, image_folder, output_dir, draw_mask=True)