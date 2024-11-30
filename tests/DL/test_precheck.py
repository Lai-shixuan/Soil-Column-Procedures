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
        cls.image_paths = [
            p for p in glob.glob('g:/DL_Data_raw/Unit_test/convert_to_float32/*') 
            if Path(p).is_file()
        ]
        
    def test_harmonized_bit_number(self, caplog):
        """Test harmonized_bit_number function for all images in one test"""
        caplog.set_level(logging.INFO)  # 设置日志级别
        logging.basicConfig(level=logging.INFO)
        
        for i, img_path in enumerate(self.image_paths):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter('ignore', DeprecationWarning)
                warnings.simplefilter("always", category=UserWarning)
                
                converted_img = precheck.harmonized_bit_number(img)
                
                if caught_warnings:
                    for warning in caught_warnings:
                        logging.info(f"Warning for image {i}: {warning.message}")
                    logging.info(f"Warning triggered for {i}, treated as success.")
                else:
                    output_path = f'g:/DL_Data_raw/Unit_test/convert_to_float32/converted/{i}.tif'
                    cv2.imwrite(output_path, converted_img)


def test_precheck():
    test_paths = fb.get_image_names('g:/DL_Data_raw/Unit_test/precheck/test_images/', None, 'png')
    test_labels_paths = fb.get_image_names('g:/DL_Data_raw/Unit_test/precheck/test_labels/', None, 'png')

    tests = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in test_paths]
    test_labels = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in test_labels_paths]
    
    datasets = precheck.precheck(tests, test_labels)

    # print every img in datasets['patches']
    for i, img in enumerate(datasets['patches']):
        cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patches/{i}.png', img)
    for i, img in enumerate(datasets['patch_labels']):
        cv2.imwrite(f'g:/DL_Data_raw/Unit_test/precheck/patch_labels/{i}.png', img)
