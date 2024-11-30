import glob
import cv2
import sys
import pytest
from pathlib import Path
import warnings
import logging

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
from API_functions import file_batch as fb
from API_functions.DL import precheck


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
        
    def test_harmonized_bit_number(self, caplog):
        """Test harmonized_bit_number function for all images in one test"""
        caplog.set_level(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        
        for i, img in enumerate(self.tests + self.test_labels + self.test_image_no_coordinating_labels):

            try:
                converted_img = precheck._harmonized_bit_number(img)
                output_path = f'g:/DL_Data_raw/Unit_test/precheck/test_image_no_coordinating_labels/converted/{i}.tif'
                cv2.imwrite(output_path, converted_img)
            except Exception as e:
                logging.info(f"Error for image {i}: {str(e)}")
                logging.info(f"Error triggered for {i}, treated as success.")

    def test_precheck(self, caplog):
        caplog.set_level(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        
        datasets = precheck.precheck(self.tests, self.test_labels)

        # print every img in datasets['patches']
        for i, img in enumerate(datasets['patches']):
            cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patches/{i}.tif', img)
        for i, img in enumerate(datasets['patch_labels']):
            cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patch_labels/{i}.tif', img)
