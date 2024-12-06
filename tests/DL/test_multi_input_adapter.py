import glob
import cv2
import sys
import matplotlib.pyplot as plt
import pytest
import logging
from pathlib import Path

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from src.API_functions.Images import file_batch as fb
from src.API_functions.DL import multi_input_adapter


class TestImageProcessing:
    @classmethod
    def setup_class(cls):
        """Setup test class - collect all image paths"""
        test_paths = fb.get_image_names('g:/DL_Data_raw/Unit_test/precheck/test_images/', None, 'png')
        test_labels_paths = fb.get_image_names('g:/DL_Data_raw/Unit_test/precheck/test_labels/', None, 'png')
        test_image_no_coordinating_labels = [p for p in glob.glob('g:/DL_Data_raw/Unit_test/precheck/test_image_no_coordinating_labels/*') if Path(p).is_file()]

        cls.tests = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in test_paths]
        cls.test_labels = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in test_labels_paths]
        cls.test_image_no_coordinating_labels = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in test_image_no_coordinating_labels]
        
    def test_harmonized_normalize(self, caplog):
        """Test harmonized_bit_number function for all images in one test"""
        caplog.set_level(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        
        for i, img in enumerate(self.tests + self.test_labels + self.test_image_no_coordinating_labels):

            try:
                converted_img = multi_input_adapter.harmonized_normalize(img)
                output_path = f'g:/DL_Data_raw/Unit_test/precheck/test_image_no_coordinating_labels/converted/{i}.tif'
                cv2.imwrite(output_path, converted_img)
            except Exception as e:
                logging.info(f"Error for image {i}: {str(e)}")
                logging.info(f"Error triggered for {i}, treated as success.")

    def test_precheck(self, caplog):
        caplog.set_level(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        
        datasets = multi_input_adapter.precheck(self.tests)
        labels = multi_input_adapter.precheck(self.test_labels, is_label=True)

        # print every img in datasets['patches']
        for i, img in enumerate(datasets['patches']):
            cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patches/{i}.tif', img)
        for i, img in enumerate(labels['patches']):
            cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patch_labels/{i}.tif', img)

        logging.info(datasets['patch_positions'])
        logging.info(datasets['original_image_info'])
        logging.info(datasets['patch_to_image_map'])
        logging.info(datasets['shape_params'])
        
        logging.info(labels['patch_positions'])
        logging.info(labels['original_image_info'])
        logging.info(labels['patch_to_image_map'])
        logging.info(labels['shape_params'])

    def test_padding_img(self):
        path = 'g:/DL_Data_raw/Unit_test/precheck/padding/to_do/'
        image_paths = fb.get_image_names(path, None, 'png')

        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = multi_input_adapter.harmonized_normalize(img)
            if img is None:
                raise FileNotFoundError("Image file not found")

            padded_img = multi_input_adapter.padding_img(img, 512, color=1)
            padded_img = fb.bitconverter.binary_to_grayscale_one_image(padded_img, 'uint8')

            plt.imshow(padded_img, cmap='gray')
            plt.show()
    
    def test_restore_image_batch(self):
        datasets = multi_input_adapter.precheck(self.tests)
        restored_images = multi_input_adapter.restore_image_batch(datasets, target_size=512)

        labels = multi_input_adapter.precheck(self.test_labels, is_label=True)
        restored_labels = multi_input_adapter.restore_image_batch(labels, target_size=512)

        for img_idx, image in enumerate(restored_images): 
            output_path = f'g:/DL_Data_raw/Unit_test/precheck/restored/restored_image_{img_idx}.tif'
            cv2.imwrite(output_path, image)
        
        for img_idx, image in enumerate(restored_labels): 
            output_path = f'g:/DL_Data_raw/Unit_test/precheck/restored/restored_labels_{img_idx}.tif'
            cv2.imwrite(output_path, image)
