import sys
sys.path.insert(0, "/root/Soil-Column-Procedures")
from API_functions.Soils import threshold_position_independent as tpi
from API_functions.DL import resize
from API_functions import file_batch as fb

import numpy as np
import cv2

def precheck(dataset: list[np.ndarray], labels: list[np.ndarray]):
    # 1. Check if dataset and labels have the same shape
    if dataset[0].shape != labels[0].shape:
        raise ValueError('Dataset and labels have different shapes')
    else:
        print('Check 1 pass, Dataset and labels have the same shape')
    
    # 2. Check if images are at least 800x800, if not, pad them. Labels and imgs have different padding color.
    check_two_pass = True
    if dataset[0].shape[0] < 800 or dataset[0].shape[1] < 800:
        dataset = [resize.padding_img(input=img, target_size=800, color=255) for img in dataset]
        print('2.Dataset images have been padded')
        check_two_pass = False
    if labels[0].shape[0] < 800 or labels[0].shape[1] < 800:
        labels = [resize.padding_img(input=img, target_size=800, color=0) for img in labels]
        print('2.Label images have been padded')
        check_two_pass = False
    if check_two_pass:
        print('Check 2 pass, Images and labels are at least 800x800')
    
    
    # 3. Check if images are 8-bit, if not, convert them
    check_three_pass = True
    if dataset[0].dtype != 'uint8':
        dataset = [fb.bitconverter.convert_to_8bit(img) for img in dataset]
        print('3.Dataset images have been converted to 8-bit')
        check_three_pass = False
    if labels[0].dtype != 'uint8':
        labels = [fb.bitconverter.convert_to_8bit(img) for img in labels]
        print('3.Label images have been converted to 8-bit')
        check_three_pass = False
    if check_three_pass:
        print('Check 3 pass, Images and labels are 8-bit') 

    
    # 4. Check if labels contain only 0 and 255
    check_four_pass = True
    for label in labels:
        unique_values = set(label.flatten())
        if unique_values != {0, 255}:
            label = tpi.user_threshold(image=label, optimal_threshold=255//2)
            check_four_pass = False
    if check_four_pass:
        print('Check 4 pass, Labels contain only 0 and 255')
    else:
        print('4.Labels have been thresholded to contain only 0 and 255')

    return dataset, labels


def test_precheck():
    test_paths = fb.get_image_names('/root/Soil-Column-Procedures/data/version1/test_images/', None, 'png')
    test_labels_paths = fb.get_image_names('/root/Soil-Column-Procedures/data/version1/test_labels/', None, 'png')

    tests = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in test_paths]
    test_labels = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in test_labels_paths]
    
    precheck(tests, test_labels)


if __name__ == '__main__':
    test_precheck()