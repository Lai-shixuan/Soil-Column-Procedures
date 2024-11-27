import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from API_functions import file_batch as fb
import os
import cv2
from tqdm import tqdm
import os
import numpy as np

# ---------------------------- 8-bit to 16-bit ----------------------------

def average(values: list):
    return sum(values) / len(values)


def convert_to_8bit(path_in, path_out):

    # Create the output directory if it does not exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Get all image files in the input directory
    image_names = fb.get_image_names(path_in, None, 'png')
    names = [os.path.basename(item) for item in image_names]
    image_files = fb.read_images(image_names, gray='gray', read_all=True)

    min_values = []
    max_values = []
    for image_file in image_files:
        min_values.append(image_file.min())
        max_values.append(image_file.max())
    my_min = average(min_values)
    my_max = average(max_values)
    
    for i, image_file in enumerate(tqdm(image_files)):
        
        # For image's value above my_max, set it to my_max
        image_file[image_file > my_max] = my_max

        # For image's value below my_min, set it to my_min
        image_file[image_file < my_min] = my_min
        image_8bit = ((image_file - my_min) / (my_max - my_min) * 255).astype(np.uint8)

        # Save the 8-bit image to the output directory
        output_path = os.path.join(path_out, names[i])
        cv2.imwrite(output_path, image_8bit)

    print("\033[1;3mConversion to 8-bit completed!\033[0m")

# ---------------------------- 16-bit to 8-bit ----------------------------
# Have 2 functions, one for converting a single image and one for batch conversion

def convert_to_16bit_one_image(image_file):
    """
    Automatically convert an image to 16-bit format, checking if the image is in 8-bit or 16-bit format.
    """
    if image_file.dtype == np.uint8:
        image_file = (image_file * 256).astype(np.uint16)
    elif image_file.dtype == np.uint16:
        print("\033[1;3mThe image is already in 16-bit format!\033[0m")
    else:
        print("\033[1;3mThe image is not in 8-bit or 16-bit format!\033[0m")
    return image_file


def convert_to_16bit(path_in: str, path_out: str, in_format: str, out_format: str='png'):
    image_names = fb.get_image_names(path_in, None, in_format)
    names = [os.path.basename(item) for item in image_names]
    image_files = fb.read_images(image_names, gray='gray', read_all=True)

    for i, image_file in enumerate(tqdm(image_files)):
        image_16bit = convert_to_16bit_one_image(image_file)

        if in_format != 'png' or in_format != 'tiff':
            names[i] = names[i].replace(in_format, out_format)

        # Save the 16-bit image to the output directory
        output_path = os.path.join(path_out, names[i])
        cv2.imwrite(output_path, image_16bit)

    print(f"\033[1;3m {i+1} images have been converted to 16-bit!\033[0m")

# ---------------------------- Test functions ----------------------------

def test_convert_to_16bit():
    path_in = 'g:/temp8bit/'
    path_out = 'g:/temp16bit/'
    convert_to_16bit(path_in, path_out, 'jpg')

# ---------------------------- Main function ----------------------------

if __name__ == '__main__':
    test_convert_to_16bit()