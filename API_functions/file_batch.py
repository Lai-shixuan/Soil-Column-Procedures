import os
import glob
import cv2
from tqdm import tqdm


# A date structure to include prefix, suffix and middle name:
class ImageName:
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix
        print(f"Your image name format is: {self.prefix}_XXXX{self.suffix}")


# A function to get all image in specific format, with prefix, and suffix:
def get_image_names(folder_path: str, my_image_names: ImageName, image_format: str):
    search_path = os.path.join(folder_path, my_image_names.prefix + '*' + my_image_names.suffix + '.' + image_format)
    image_files_names = glob.glob(search_path)

    # Test if there is any image in the folder:
    if not image_files_names:
        raise Exception('Error: No images found')

    # Some information about the images:
    print(f"{len(image_files_names)} images have been found in {folder_path}")
    if len(image_files_names) > 3:
        print("The first 3 images are:")
        for i in range(3):
            print(image_files_names[i])
    else:
        print("All images are:")
        for image_file in image_files_names:
            print(image_file)
    if len(image_files_names) > 1000:
        print("\033[1;3mWarning\033[0m, your files are too large to read all.")
    print(f"\033[1;3mGet names completely!\033[0m")
    return image_files_names


def read_images(image_files_names: list, gray: bool = True, read_all: bool = False, read_num: int = 1000):
    """
    By default, not read all images. If you want to read all images, please set read_all=True,
    and delete read_num parameter.
    """
    if not image_files_names:
        raise Exception('Error: No images found')
    images = []
    if read_all:
        for image_file in tqdm(image_files_names):
            if gray:
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_file)
            images.append(image)
        print(f"{len(images)} images have been read")
        print(f"\033[1;3mReading completely!\033[0m")
        return images
    else:
        for image_file in tqdm(image_files_names[:min(read_num, len(image_files_names))]):
            if gray:
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_file)
            images.append(image)
        print(f"first {len(images)} images have been read")
        print(f"if you want to read all, please set read_all=True")
        print(f"\033[1;3mReading completely!\033[0m")
        return images


# A function to output all images to a specific folder in a specific format, with a specific name format:
def output_images(image_files: list, output_folder: str, my_image_name: ImageName, output_format: str):
    if not image_files:
        raise Exception('Error: No images found')
    for idx, image_file in enumerate(tqdm(image_files)):
        new_file_name = f"{my_image_name.prefix}_{idx:04d}{my_image_name.suffix}.{output_format}"
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
        if my_image_name.prefix != '':
            new_file_name = f"{my_image_name.prefix}_{idx:04d}{my_image_name.suffix}.{output_format}"
        else:
            new_file_name = f"{idx:04d}{my_image_name.suffix}.{output_format}"
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
