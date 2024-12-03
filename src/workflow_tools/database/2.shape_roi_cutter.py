import cv2
import numpy as np
import sys

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from tqdm import tqdm
from src.API_functions.Images import file_batch as fb
from API_functions.Soils import threshold_position_independent as tpi
from API_functions.DL import shape_processor as processor
from API_functions.DL import shape_detectors as detector
from API_functions.DL import multi_input_adapter as adapter

class ShapeROICutter:
    """Utility class for cutting ROIs from images based on shape detection in label images"""
    
    @staticmethod
    def save_processed_results(processed_image: np.ndarray, output_dir: str, base_name: str, suffix: str = ''):
        """Helper function to save processed images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{base_name}{suffix}.tif"
        cv2.imwrite(str(output_path), processed_image)

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
        
        # Process all images
        for i in tqdm(range(len(label_images_files)), desc="Processing images"):
            label_image = label_images_files[i]
            image = images_files[i]
            base_name = Path(label_paths[i]).stem
            
            # Detect shape parameters from label, and cut image
            label_images, params = processor.process_shape_detection(
                label_image, 
                detector.EllipseDetector(), 
                is_label=True, 
                draw_mask=draw_mask
            )
            image_cut = processor.cut_with_shape(image, params, fillcolor=fill_color)

            # Save results
            cls.save_processed_results(label_images['cut'], output_dir / 'label', base_name)
            cls.save_processed_results(image_cut, output_dir / 'image', base_name)

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
    
    cutter = ShapeROICutter()
    cutter.process_folder(label_folder, image_folder, output_dir)