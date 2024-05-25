import os
import glob
from typing import Union, Optional
import cv2
from tqdm import tqdm
from API_functions.column_batch_process import ImageName


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
def get_image_names(folder_path: str, my_image_names: Union[ImageName, None], image_format: str):
    if my_image_names is not None:
        my_image_names.prefix = my_image_names.prefix[:-1]
        my_image_names.suffix = my_image_names.suffix[1:]
        search_path = os.path.join(folder_path, my_image_names.prefix + '*' + my_image_names.suffix + '.' + image_format)
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


# A function to read all images in specific format, with gray, turn to gray, or color:
def read_images(image_files_names: list, gray: str = "gray", read_all: bool = False, read_num: int = 1000):
    """
    By default, not read all images. If you want to read all images, please set read_all=True,
    and delete read_num parameter.
    """
    def read():
        if gray == "gray":
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
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
def output_images(image_files: list, output_folder: str, my_image_name: ImageName, output_format: str):
    if not image_files:
        raise Exception('Error: No images found')
    for idx, image_file in enumerate(tqdm(image_files)):
        new_file_name = f"{my_image_name.prefix}{idx:05d}{my_image_name.suffix}.{output_format}"
        new_file_path = os.path.join(output_folder, new_file_name)
        cv2.imwrite(new_file_path, image_file)
    print(f"{len(image_files)} images have been saved to {output_folder} with the format {output_format}")
    print("\033[1;3mOutput completely!\033[0m")


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


# A function to convert to specific format
def format_conversion(folder_path, input_image_name, output_image_name, input_format, output_format):
    image_names = get_image_names(folder_path, input_image_name, format)


# Function to convert BMP images to PNG format
# def convert_to_(folder_path):
#     search_path = os.path.join(folder_path, '*.bmp')
#     bmp_files = glob.glob(search_path)
#
#     for idx, bmp_file in enumerate(bmp_files):
#         image = cv2.imread(bmp_file, cv2.IMREAD_GRAYSCALE)
#         new_file_name = f"1_png_{idx:03d}.png"
#         new_file_path = os.path.join(folder_path, new_file_name)
#         cv2.imwrite(new_file_path, image)
#
#
# # Function to convert images to binary format
# def convert_to_binary(folder_path):
#     search_path = os.path.join(folder_path, '*.png')
#     png_files = glob.glob(search_path)
#
#     for idx, png_file in enumerate(png_files):
#         image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
#         if image is not None:
#             binary_image = image // 255
#             new_file_name = f"2_binary_{idx:03d}.png"
#             new_file_path = os.path.join(folder_path, new_file_name)
#             cv2.imwrite(new_file_path, binary_image)
#         else:
#             print(f"Image not found: {png_file}")


# Function to convert binary images to grayscale
def binary_to_grayscale(folder_path):
    search_path = os.path.join(folder_path, '*.png')
    binary_files = glob.glob(search_path)

    for idx, binary_file in enumerate(binary_files):
        image = cv2.imread(binary_file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            grayscale_image = image * 255
            new_file_name = f"3_grayscale_{idx:03d}.png"
            new_file_path = os.path.join(folder_path, new_file_name)
            cv2.imwrite(new_file_path, grayscale_image)
        else:
            print(f"Image not found: {binary_file}")
