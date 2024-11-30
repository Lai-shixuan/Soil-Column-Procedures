import os
import glob
from typing import Union, Optional
import cv2
from tqdm import tqdm
import numpy as np
import shutil

import sys
# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
from API_functions.Soils import pre_process
from API_functions.Soils import threshold_position_independent


# A date structure to include prefix, suffix and middle name:
class ImageName:
    def __init__(self, prefix, suffix):
        if prefix == '':
            self.prefix = ''
        else:
            self.prefix = prefix + '_'
        if suffix == '':
            self.suffix = ''
        else:
            self.suffix = '_' + suffix
        print(f"Your image name format is: {self.prefix}XXXX{self.suffix}")


# A class to define the region of interest:
class roi_region:
    def __init__(self, x1, y1, width, height, z1, z2):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.z2 = z2
        self.width = width
        self.height = height
        self.depth = abs(z2 - z1) + 1
        # check roi information is complete
        if x1 is None or y1 is None or width is None or height is None or z1 is None or z2 is None:
            print("Warning: Please check your ROI information is complete or not.")
        print(f"Your ROI is: x1={x1}, y1={y1}, width={width}, height={height}, depth={z1}-{z2}={self.depth}")


# --------------------------------- get_image_names --------------------------------- #

# A function to show the image names:
def show_image_names(names):
    if len(names) > 3:
        print("The first 3 images are:")
        for i in range(3):
            print(names[i])
    else:
        print("All images are:")
        for image_file in names:
            print(image_file)
    if len(names) > 1000:
        print("\033[1;3mWarning\033[0m, your files are too large to read all.")


# A function to get all image in specific format, with prefix, and suffix:
def get_image_names(folder_path: str, image_names: Union[ImageName, None], image_format: str):
    if image_names is not None:
        image_names.prefix = image_names.prefix[:-1]
        image_names.suffix = image_names.suffix[1:]
        search_path = os.path.join(folder_path, image_names.prefix + '*' + image_names.suffix + '.' + image_format)
    else:
        search_path = os.path.join(folder_path, '*.' + image_format)
    image_files_names = glob.glob(search_path)

    # Test if there is any image in the folder:
    if not image_files_names:
        raise Exception('Error: No images found')

    # Some information about the images:
    print(f"{len(image_files_names)} images have been found in {folder_path}")
    show_image_names(image_files_names)
    print(f"\033[1;3mGet names completely!\033[0m")
    return image_files_names

# --------------------------------- read_images and output_images, huge rom needed --------------------------------- #

# A function to read all images in specific format, with gray, turn to gray, or color:
def read_images(image_files_names: list, gray: str = "gray", read_all: bool = False, read_num: int = 1000):
    """
    By default, not read all images. If you want to read all images, please set read_all=True,
    and delete read_num parameter.
    """
    def read():
        if gray == "gray":
            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        elif gray == "turn to gray":
            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif gray == "color":
            image = cv2.imread(image_file)
        else:
            raise Exception('Error: Please set gray to "gray", "turn to gray" or "color"')
        images.append(image)

    if not image_files_names:
        raise Exception('Error: No images found')
    images = []
    if read_all:
        for image_file in tqdm(image_files_names):
            read()
        print(f"{len(images)} images have been read")
        print(f"\033[1;3mReading completely!\033[0m")
        return images
    else:
        for image_file in tqdm(image_files_names[:min(read_num, len(image_files_names))]):
            read()
        print(f"first {len(images)} images have been read")
        print(f"if you want to read all, please set read_all=True")
        print(f"\033[1;3mReading completely!\033[0m")
        return images


# A function to output all images to a specific folder in a specific format, with a specific name format:
def output_images(image_files: np.ndarray, output_folder: str, my_image_name: ImageName, output_format: str):
    if not image_files:
        raise Exception('Error: No images found')
    for idx, image_file in enumerate(tqdm(image_files)):
        new_file_name = f"{my_image_name.prefix}{idx:05d}{my_image_name.suffix}.{output_format}"
        new_file_path = os.path.join(output_folder, new_file_name)
        cv2.imwrite(new_file_path, image_file)
    print(f"{len(image_files)} images have been saved to {output_folder} with the format {output_format}")
    print("\033[1;3mOutput completely!\033[0m")

# --------------------------------- format_transformer --------------------------------- #

# A function to read image one by one and output as specific format and name
def format_transformer(image_name_lists: list[str], output_folder: str,
                       my_image_name: ImageName, output_format: str,
                       read_all: bool = True, read_num: int = 1000):
    """
    Only for gray images!
    It read an image and write it now so that it will not occupy too much memory.
    set read_all=False, and set read_num to a specific number if you do not want to transformer all images.
    """
    def read_and_write():
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        new_file_name = f"{my_image_name.prefix}{idx:05d}{my_image_name.suffix}.{output_format}"
        new_file_path = os.path.join(output_folder, new_file_name)
        cv2.imwrite(new_file_path, image)

    if read_all:
        print(f"if you do not want to transformer all images,"
              f" please set read_all=False, and set read_num to a specific number.")
        for idx, image_name in enumerate(tqdm(image_name_lists)):
            read_and_write()
    else:
        for idx, image_name in enumerate(tqdm(image_name_lists[:min(read_num, len(image_name_lists))])):
            read_and_write()

    print(f"first {min(read_num, len(image_name_lists))} images"
          f" have been saved to {output_folder} with the format {output_format}")
    print("\033[1;3mOutput completely!\033[0m")



# Function to convert binary images to grayscale
def binary_to_grayscale(read_path:str, image_names:Union[ImageName, None], format:str, output_path:str):

    png_files = get_image_names(folder_path=read_path, image_names=image_names, image_format=format)
    for idx, binary_file in enumerate(png_files):
        image = cv2.imread(binary_file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            grayscale_image = image * 255
            new_file_path = os.path.join(output_path, os.path.basename(binary_file))
            cv2.imwrite(new_file_path, grayscale_image)
        else:
            print(f"Image not found: {binary_file}")


# Function to convert grayscale images to RGB
def grayscale_to_rgb(read_path, output_path):
    png_files = get_image_names(folder_path=read_path, image_names=None, image_format='png')

    for idx, png_file in enumerate(png_files):
        image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if image is not None:
            new_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            new_file_path = os.path.join(output_path, os.path.basename(png_file))
            cv2.imwrite(new_file_path, new_image)
    print("\033[1;3mConversion to RGB completed!\033[0m") 


# Convert RGB images to grayscale
def rgb_to_grayscale(read_path, output_path):
    png_files = get_image_names(folder_path=read_path, image_names=None, image_format='png')

    for idx, png_file in enumerate(png_files):
        image = cv2.imread(png_file)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if image is not None:
            new_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            new_file_path = os.path.join(output_path, os.path.basename(png_file))
            cv2.imwrite(new_file_path, new_image)
    print("\033[1;3mConversion to grayscale completed!\033[0m")


class bitconverter:
    """
    A class to convert images between 8-bit and 16-bit formats.
    """
    def __init__(self):
        pass


    def average(values: list):
        return sum(values) / len(values)

    # ---------------------------- 16-bit to 8-bit ----------------------------

    def convert_to_8bit_one_image(image_file, type='float'):
        if image_file.dtype == np.uint16:
            image_8bit = (image_file / 256)
            if type == 'float':
                image_8bit = image_8bit.astype(np.float32)
            elif type == 'uint8':
                image_8bit = image_8bit.astype(np.uint8)
        elif image_file.dtype == np.uint8:
            print("\033[1;3mThe image is already in 8-bit format!\033[0m")
        else:
            print("\033[1;3mThe image is not in 8-bit or 16-bit format!\033[0m")
        return image_8bit


    def convert_to_8bit(path_in, path_out, in_format: str):
        """
        Only for grayscale images.
        Optional between float and uint8.
        If it need windows adjustment, please use the other function first in file batch.
        """
        # Create the output directory if it does not exist
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # Get all image files in the input directory
        image_names = get_image_names(path_in, None, in_format)
        names = [os.path.basename(item) for item in image_names]
        image_files = read_images(image_names, gray='gray', read_all=True)

        for i, image_file in enumerate(tqdm(image_files)):

            image_8bit = bitconverter.convert_to_8bit_one_image(image_file, type='uint8')
            output_path = os.path.join(path_out, names[i])
            cv2.imwrite(output_path, image_8bit)
        print(f"\033[1;3m{i+1} images have been converted to 8-bit!\033[0m")

    # ---------------------------- 8-bit to 16-bit ----------------------------
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


    def convert_to_16bit(path_in: str, path_out: str, in_format: str, out_format: str):
        # Create the output directory if it does not exist
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        image_names = get_image_names(path_in, None, in_format)
        names = [os.path.basename(item) for item in image_names]
        image_files = read_images(image_names, gray='gray', read_all=True)

        for i, image_file in enumerate(tqdm(image_files)):
            image_16bit = bitconverter.convert_to_16bit_one_image(image_file)

            if in_format != 'png' or in_format != 'tiff':
                names[i] = names[i].replace(in_format, out_format)

            # Save the 16-bit image to the output directory
            output_path = os.path.join(path_out, names[i])
            cv2.imwrite(output_path, image_16bit)

        print(f"\033[1;3m {i+1} images have been converted to 16-bit!\033[0m")
    
    # ---------------------------- Grayscale to binary ----------------------------
    # Have 2 functions, one for converting a single image and one for batch conversion

    def grayscale_to_binary_one_image(image):
        """
        Convert a single grayscale image to binary.
        """
        if image.dtype == np.uint8:
            binary_image = (image / 255).astype(np.float32)
        elif image.dtype == np.uint16:
            binary_image = (image / 65535).astype(np.float32)
        elif image.dtype == np.float32:
            if np.max(image) <= 1:
                binary_image = image
            elif np.max(image) <= 255:
                binary_image = image / 255
            elif np.max(image) <= 65535:
                binary_image = image / 65535
        else:
            raise Exception('Error: Image is not in 8-bit or 16-bit or float32 grayscale format!')
        return binary_image


    def grayscale_to_binary(read_path: str, in_format: str, read_name: Union[ImageName, None], output_path: str):
        """
        Don't use this function, because please note that the output format have to be tiff to save the binary images from 0 to 1.
        Function to convert images to binary format.
        """
        image_paths = get_image_names(folder_path=read_path, image_names=read_name, image_format=in_format)

        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if image is not None:
                binary_image = bitconverter.grayscale_to_binary_one_image(image)
                # change in_format to tiff:
                file_name = os.path.basename(image_path).replace(in_format, 'tiff')
                new_file_path = os.path.join(output_path, file_name)
                cv2.imwrite(new_file_path, binary_image)

        print(f"\033[1;3m{len(image_paths)} images' conversion to binary completed!\033[0m")


# ---------------------------- Test functions ----------------------------

def test_convert_to_8bit():
    path_in = 'g:/temp16bit/'
    path_out = 'g:/temp8bit/'
    bitconverter.convert_to_8bit(path_in, path_out, 'png')


def test_convert_to_16bit():
    path_in = 'g:/temp8bit/'
    path_out = 'g:/temp16bit/'
    bitconverter.convert_to_16bit(path_in, path_out, 'jpg')


def test_grayscale_to_binary():
    path_in = 'g:/DL_Data_raw/Unit_test/grayscale_to_binary/16bit/'
    path_out = 'g:/DL_Data_raw/Unit_test/grayscale_to_binary/0-1/'
    bitconverter.grayscale_to_binary(path_in, 'png', None, path_out)

# --------------------------------- column_batch_related --------------------------------- #


# Crop the image without name change, but the folder will change, change the format to png
def roi_select(path_in: str, path_out: str, name_read: Union[ImageName, None], roi: roi_region, img_format: str = 'bmp'):
    """
    Only for gray images!
    The list of names will not change, but the folder will change.
    The format will change to png.
    For extract_index function within the function, it only works for filenames with '1-1_rec00000105.bmp' or '0029.001.png'
    """

    def extract_index(filename):
        # Split the filename to isolate the numeric part
        # Only works for filenames with '1-1_rec00000105.bmp'
        if 'rec' in filename:
            parts = filename.split('_rec')
            if len(parts) > 1:
                numeric_part = parts[1].split('.')[0]
                return int(numeric_part)  # Convert to int to strip leading zeros
            return None
        # Only works for filenames with '0029.001.png'
        # If using '.' to split, be careful with the address which also contains '.'
        elif '.' in filename:
            parts = filename.split('.')
            if len(parts) > 1:
                numeric_part = parts[-2]
                return int(numeric_part)

    image_files = get_image_names(folder_path=path_in, image_names=name_read, image_format=img_format)
    if not image_files:
        raise Exception('Error: No images found')
    
    # Extract_index from file names and paths, like [15, 16, ..., 3311]
    indexes = [extract_index(image) for image in image_files]

    # Find the index of user givern z1 and z2 in the indexes list. Elements can not be repeated.
    # This is necessary because the file may not start from 0 or 1.
    z1_index = indexes.index(roi.z1)
    z2_index = indexes.index(roi.z2)
    if z1_index > z2_index:
        z1_index, z2_index = z2_index, z1_index

    # Filter the image_files with z1 and z2 index
    image_files = image_files[z1_index: z2_index+1]
    
    # cut the roi region
    temp_list = []
    for image_file in tqdm(image_files):

        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        roi_image = image[roi.y1:roi.y1 + roi.height, roi.x1:roi.x1 + roi.width]
        old_file_name = os.path.basename(image_file)
        old_file_name, _ = os.path.splitext(old_file_name)
        new_file_path = os.path.join(path_out, old_file_name + '.png')
        cv2.imwrite(new_file_path, roi_image)
        temp_list.append(new_file_path)
    show_image_names(temp_list)
    print("\033[1;3mROI Selected Completely!\033[0m")


# Rename the image with new name, but the folder will change, only os operation
def rename(path_in: str, path_out: str, new_name: ImageName, reverse: bool, start_index: int = 1, overwrite: bool=False):
    """
    Only for gray images!
    You can not change image format.
    The list of names will change to the new name.
    """

    image_files = get_image_names(folder_path=path_in, image_names=None, image_format='png')
    if not image_files:
        raise Exception('Error: No images found')
    if reverse:
        image_files.reverse()

    namelist_new = []
    _, extension = os.path.splitext(image_files[0])
    extension = extension[1:]
    
    for filename in tqdm(image_files):
        
        new_file_name = f'{new_name.prefix}{start_index:05d}{new_name.suffix}.{extension}'
        new_file = os.path.join(path_out, new_file_name)

        # detect whether has a file in that path
        if os.path.exists(new_file):
            if not overwrite:
                raise Exception('Error: The file has existed, please change the name or set overwirte mode.')
            else:
                os.remove(new_file)

        shutil.copy2(filename, new_file)
        namelist_new.append(new_file)
        start_index += 1

    # Clear the old list and add the new list:
    show_image_names(namelist_new)
    print(f'\033[1;3mRename completely!\033[0m')


def windows_adjustment(path_in: str, path_out: str):
    """
    Only for gray images! png format.
    """
    image_lists = get_image_names(path_in, None, 'png')
    image_names = [os.path.basename(item) for item in image_lists]
    images = read_images(image_lists, gray='gray', read_all=True)
    min, max = pre_process.get_average_windows(images)
    print(f"min: {min}, max: {max}")

    for j, image in enumerate(tqdm(images)):
        image = pre_process.map_range_to_65536(image, min, max)

        if not os.path.exists(path_out):
            os.makedirs(path_out)
        cv2.imwrite(os.path.join(path_out, image_names[j]), image)
    print(f'\033[1;3mWindows adjustment completely!\033[0m')


# Threshold the image with new name, but the folder will change
def image_process(path_in: str, path_out: str):
    """
    Only for gray scale image.
    No name change.
    """
    
    image_files = get_image_names(folder_path=path_in, image_names=None, image_format='png')
    if not image_files:
        raise Exception('Error: No images found')
    
    temp_list = []

    for image_name in tqdm(image_files):
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        image_prepropossed = pre_process.origin(image)
        image_threshold = threshold_position_independent.user_threshold(image_prepropossed, 145)
        image_invert = cv2.bitwise_not(image_threshold)     # invert image, make the pore space to be white

        # save with the same name but change folder:
        save = os.path.join(path_out, os.path.basename(image_name))
        
        cv2.imwrite(save, image_invert)
        temp_list.append(save)

    show_image_names(temp_list)
    print(f'\033[1;3mImage procession completely!\033[0m')


# ---------------------------- Main function ----------------------------

if __name__ == '__main__':
    test_grayscale_to_binary()
