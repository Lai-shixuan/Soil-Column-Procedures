import pytest
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from API_functions import file_batch as fb
from API_functions.DL import shape_processor as processor
from API_functions.DL import shape_detectors as detector
from API_functions.DL import multi_input_adapter as adapter


class TestHyperRoiCutted:
    def setup_method(self):

        # ---------------------------- Cut with Shape ---------------------------- #

        # Define common paths used for special cases
        self.label_folder = 'g:/DL_Data_raw/Unit_test/round_square/labels/'
        self.image_folder = 'g:/DL_Data_raw/Unit_test/round_square/images/'
        # self.special_output_dir = "g:/DL_Data_raw/Unit_test/round_square/special_output_dir/"
        self.label_folder = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/labels/'
        self.image_folder = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/1.Reconstruct/images/'
        self.special_output_dir = 'f:/3.Experimental_Data/Soils/Online/Soil.column.0035/2.ROI/'
        
        # Load all images for both tests
        self.label_paths = fb.get_image_names(self.label_folder, None, 'png')
        self.image_paths = fb.get_image_names(self.image_folder, None, 'png')
        
        # Read all images
        self.label_images = [adapter.harmonized_normalize(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)) for path in self.label_paths]
        self.images = [adapter.harmonized_normalize(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)) for path in self.image_paths]

        # ---------------------------- Batch processing ---------------------------- #

        # Define common paths used for batch processing
        self.batch_label_folder = 'g:/DL_Data_raw/Unit_test/round_square_inference/labels/'
        self.batch_image_folder = 'g:/DL_Data_raw/Unit_test/round_square_inference/images_cutted/'
        self.batch_output_dir = "g:/DL_Data_raw/Unit_test/round_square_inference/batch/"

        # Load all images for batch processing
        self.batch_label_paths = fb.get_image_names(self.batch_label_folder, None, 'png')
        self.batch_image_paths = fb.get_image_names(self.batch_image_folder, None, 'png')

        # Read all images for batch processing
        self.batch_label_images = [adapter.harmonized_normalize(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)) for path in self.batch_label_paths]
        self.batch_images = [adapter.harmonized_normalize(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)) for path in self.batch_image_paths]

    @staticmethod
    def save_processed_results(processed_image: np.ndarray, output_dir: str, base_name: str, suffix: str = ''):
        """Helper function to save processed images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{base_name}{suffix}.tif"
        cv2.imwrite(str(output_path), processed_image)

    def test_hyper_roi_cutted_batch(self):
        """Test original batch processing"""

        # Manully set parameters
        draw_mask = True

        output_dir = Path(self.batch_output_dir)
        
        # Process all label images
        for i in tqdm(range(len(self.batch_label_images))):

            base_name = Path(self.batch_label_paths[i]).stem
            
            # Process label
            label_image = self.batch_label_images[i]
            label_result = processor.process_shape_detection(label_image, detector.EllipseDetector(), is_label=True, draw_mask=draw_mask)
            label_images, params = label_result
            
            # Save label cut result
            self.save_processed_results(label_images['cut'],  output_dir / 'labels', base_name)
            
            
            # Process corresponding image
            image = self.batch_images[i]
            image_result = processor.process_shape_detection(image, detector.EllipseDetector(), is_label=False, draw_mask=draw_mask)
            image_images, _ = image_result
            
            # Save image cut result
            self.save_processed_results(image_images['cut'], output_dir / 'images', base_name)
            
            # Save draws result if available
            if 'draw' in image_images:
                self.save_processed_results(image_images['draw'], output_dir / 'images_draw', base_name)
            if 'draw' in label_images:
                self.save_processed_results(label_images['draw'], output_dir / 'labels_draw', base_name)

    def test_cut_with_shape(self):
        """
        Test new parameter reuse functionality in batch mode.
        Process all label images to detect parameters, then apply them to corresponding dataset images.
        """
        # Manually set parameters
        is_label = True
        fill_color = 0
        draw_mask = True

        output_dir = Path(self.special_output_dir)
        
        # Process all images
        for i in tqdm(range(len(self.label_images))):
            # Get corresponding images and base name
            label_image = self.label_images[i]
            image = self.images[i]
            base_name = Path(self.label_paths[i]).stem
            
            # Detect shape parameters from label, and cut image
            label_images, params = processor.process_shape_detection(label_image, detector.EllipseDetector(), is_label=is_label, draw_mask=draw_mask)
            image_cut = processor.cut_with_shape(image, params, fillcolor=fill_color)

            # Save results
            self.save_processed_results(label_images['cut'], output_dir / 'label', base_name)
            self.save_processed_results(image_cut, output_dir / 'image', base_name)

            if draw_mask:
                # Draw dataset image with detected parameters
                image = fb.bitconverter.binary_to_grayscale_one_image(image, 'uint8')
                image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image_draw = processor.ShapeDrawer.draw_shape(image_draw, params)
                
                # Save results 
                self.save_processed_results(label_images['draw'], output_dir / 'label_draw', base_name)
                self.save_processed_results(image_draw, output_dir / 'image_draw', base_name)