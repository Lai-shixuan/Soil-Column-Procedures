# Conmon library import
import os
import cv2
from tqdm import tqdm
import shutil

# User-defined library import
from . import file_batch
from . import pre_process
from . import threshold_position_independent


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
        print(f"Your ROI is: x1={x1}, y1={y1}, width={width}, height={height}, depth={z1}-{z2}")


# A class to define the soil column structure, including parts and subfolders:
class SoilColumn:

    # A class to define the part structure, including part_id and subfolders:
    class Part:
        def __init__(self, part_id, part_path):
            self.part_id = part_id
            self.path = part_path
            self.sub_folders = [
                os.path.join(part_path, "0.Origin"),
                os.path.join(part_path, "1.Reconstruct"),
                os.path.join(part_path, "2.ROI"),
                os.path.join(part_path, "3.Rename"),
                os.path.join(part_path, "4.Threshold"),
                os.path.join(part_path, "5.Analysis")
            ]

        def get_subfolder_path(self, index):
            if 0 <= index < len(self.sub_folders):
                return self.sub_folders[index]
            return None

    def __init__(self, root_path):
        self.root_path = root_path
        self.id = self._extract_id(root_path)
        self.parts = self._load_parts()

    def _extract_id(self, path):
        # Extract the last part of the path and split it by '.'
        base_name = os.path.normpath(path).split(os.sep)[-1]
        if base_name.startswith("Soil.column."):
            return base_name.split(".")[-1]
        return None

    def _load_parts(self):
        parts = []
        for part_name in os.listdir(self.root_path):
            part_path = os.path.join(self.root_path, part_name)
            if os.path.isdir(part_path) and part_name.startswith(self.id):
                part_id = part_name.split("-")[-1]
                parts.append(SoilColumn.Part(part_id, part_path))
        return parts

    def get_part(self, part_index):
        for part in self.parts:
            if int(part.part_id) == part_index:
                return part
        return None

    def get_subfolder_path(self, part_index, folder_index):
        part = self.get_part(part_index)
        if part:
            return part.get_subfolder_path(folder_index)
        return None


# Create the column structure with parts and subfolders
def create_column_structure(column_id, part_ids, base_path):
    # Ensure column_id is 4 digits
    column_id = f"Soil.column.{int(column_id):04d}"
    column_path = os.path.join(base_path, column_id)
    os.makedirs(column_path, exist_ok=True)
    
    for part_id in part_ids:
        # Ensure part_id is 2 digits
        part_id = f"{int(part_id):02d}"
        part_path = os.path.join(column_path, f"{column_id.split('.')[-1]}-{part_id}")
        os.makedirs(part_path, exist_ok=True)
        
        sub_folders = [
            "0.Origin",
            "1.Reconstruct",
            "2.ROI",
            "3.Rename",
            "4.Threshold",
            "5.Analysis"
        ]
        
        for folder in sub_folders:
            os.makedirs(os.path.join(part_path, folder), exist_ok=True)

    return column_path


# Crop the image without name change, but the folder will change, change the format to png
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
    file_batch.show_image_names(temp_list)
    print("\033[1;3mROI Selected Completely!\033[0m")
    return temp_list


# Rename the image with new name, but the folder will change, change the format to png
def rename(image_files: list[str], new_path: str, new_name: ImageName, start_index: int = 1, overwrite: bool=False):
    """
    You can not change image format.
    The list of names will change to the new name.
    """

    namelist_new = []
    _, extension = os.path.splitext(image_files[0])
    extension = extension[1:]
    
    for filename in tqdm(image_files):
        
        new_file_name = f'{new_name.prefix}{start_index:05d}{new_name.suffix}.{extension}'
        new_file = os.path.join(new_path, new_file_name)

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
    file_batch.show_image_names(namelist_new)
    print(f'\033[1;3mRename completely!\033[0m')
    return namelist_new


# Threshold the image with new name, but the folder will change, change the format to png
def image_process(namelists: list, save_path: str, save_name: ImageName, start_index: int = 1):
    """
    Only for gray scale image.
    """
    
    temp_list = []

    for name in tqdm(namelists):
        image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        # image = fs.read_image_opencv(str(os.path.join(batch_read_path, name)))
        image_prepropossed = pre_process.median(image, 5)
        image_threshold = threshold_position_independent.user_threshold(image_prepropossed, 120)
        image_invert = cv2.bitwise_not(image_threshold)     # invert image, make the pore space to be white

        save = os.path.join(save_path, f'{save_name.prefix}{str(start_index).zfill(5)}{save_name.suffix}.png')
        start_index += 1
        
        cv2.imwrite(save, image_invert)
        temp_list.append(save)

    file_batch.show_image_names(temp_list)
    return temp_list