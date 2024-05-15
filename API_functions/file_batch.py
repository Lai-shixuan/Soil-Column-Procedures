import os
import glob
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm


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


class roi_region:
    def __init__(self, x1, y1, width, height, z1, z2):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.z2 = z2
        self.width = width
        self.height = height
        self.depth = abs(z2 - z1) + 1
        print(f"Your ROI is: x1={x1}, y1={y1}, width={width}, height={height}, depth={z1}-{z2}")


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
def get_image_names(folder_path: str, my_image_names: ImageName, image_format: str):
    my_image_names.prefix = my_image_names.prefix[:-1]
    my_image_names.suffix = my_image_names.suffix[1:]
    search_path = os.path.join(folder_path, my_image_names.prefix + '*' + my_image_names.suffix + '.' + image_format)
    image_files_names = glob.glob(search_path)

    # Test if there is any image in the folder:
    if not image_files_names:
        raise Exception('Error: No images found')

    # Some information about the images:
    print(f"{len(image_files_names)} images have been found in {folder_path}")
    show_image_names(image_files_names)
    print(f"\033[1;3mGet names completely!\033[0m")
    return image_files_names


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


def create_nifti(image_lists: list[str], output_folder: str, nifti_name: str):
    """
    Only for gray images!
    It will read all images and stack them together, which will take up a lot of memory.
    """
    images = read_images(image_lists, gray="gray", read_all=True)
    combined_array = np.stack(images, axis=-1)
    nifti_img = nib.Nifti1Image(combined_array, affine=np.eye(4))
    output_path = os.path.join(output_folder, nifti_name + '.nii') 

    # Detect whether has a nib file in that path
    if os.path.exists(output_path):
        raise Exception('Error: The file has existed, please change the name.')

    nib.save(nifti_img, output_path)
    print("\033[1;3mSave Done!\033[0m")


def rename(image_files: list[str], new_name: ImageName, start_index: int = 1, overwrite: bool=False):
    """
    You can not change image format.
    The list of names will change to the new name.
    """

    namelist_new = []
    _, extension = os.path.splitext(image_files[0])
    extension = extension[1:]
    folder = os.path.dirname(image_files[0])
    
    for filename in tqdm(image_files):
        
        new_file_name = f'{new_name.prefix}{start_index:05d}{new_name.suffix}.{extension}'
        new_file = os.path.join(folder, new_file_name)

        # detect whether has a file in that path
        if os.path.exists(new_file):
            if not overwrite:
                raise Exception('Error: The file has existed, please change the name or set overwirte mode.')
            else:
                os.remove(new_file)

        os.rename(filename, new_file)
        namelist_new.append(new_file)
        start_index += 1

    # Clear the old list and add the new list:
    show_image_names(namelist_new)
    print(f'\033[1;3mRename completely!\033[0m')
    return namelist_new


# crop the image without name change, but the folder will change, change the format to png
def roi_select(image_files: list[str], path: str, roi: roi_region):
    """
    The list of names will not change, but the folder will change.
    The format will change to png.
    Only for gray images!
    """
    
    def extract_index(filename):
        # Split the filename to isolate the numeric part
        parts = filename.split('_rec')
        if len(parts) > 1:
            numeric_part = parts[1].split('.')[0]
            return int(numeric_part)  # Convert to int to strip leading zeros
        return None

    if not image_files:
        raise Exception('Error: No images found')
    
    # Extract_index from file lists, like [15, 16, ..., 3311]
    indexes = [extract_index(image) for image in image_files]

    # Find the index of user givern z1 and z2 in the indexes list. Elements can not be repeated.
    z1_index = indexes.index(roi.z1)
    z2_index = indexes.index(roi.z2)
    if z1_index > z2_index:
        z1_index, z2_index = z2_index, z1_index

    # Filter the image_files with z1 and z2 index
    image_files = image_files[z1_index: z2_index+1]
    
    temp_list = []
    for image_file in tqdm(image_files):

        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        roi_image = image[roi.y1:roi.y1 + roi.height, roi.x1:roi.x1 + roi.width]
        old_file_name = os.path.basename(image_file)
        old_file_name, _ = os.path.splitext(old_file_name)
        new_file_path = os.path.join(path, old_file_name + '.png')
        cv2.imwrite(new_file_path, roi_image)
        temp_list.append(new_file_path)
    show_image_names(temp_list)
    print("\033[1;3mROI Selected Completely!\033[0m")
    return temp_list