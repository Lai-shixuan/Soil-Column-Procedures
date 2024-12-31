import cv2
import numpy as np
import random
import sys

sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from skimage import measure
from pathlib import Path
from typing import Tuple, List
from src.workflow_tools.database.s4refine_label import read_tif_image_label
from src.API_functions.Images import file_batch as fb

class ImageAugmenter:
    """A class for performing image augmentation on data and label image pairs.

    This class implements various image augmentation techniques including flipping,
    rotation, and object relocation.

    Args:
        data_img (np.ndarray): The input data image to be augmented.
        label_img (np.ndarray): The corresponding label image to be augmented.

    Attributes:
        data_img (np.ndarray): The stored data image.
        label_img (np.ndarray): The stored label image.
        labeled_img (np.ndarray): Connected component labeled version of label_img.
        num_objects (int): Number of distinct objects in the label image.
        background_value (float): Mean pixel value of the background in data_img.
    """

    def __init__(self, data_img: np.ndarray, label_img: np.ndarray):
        """Initialize augmenter with data and label images"""
        self.data_img = data_img
        self.label_img = label_img
        self.labeled_img = measure.label(label_img)
        self.num_objects = np.max(self.labeled_img)
        self.background_value = np.mean(data_img[label_img == 0])
        
    def _extract_object(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extracts an object from an image using its mask.

        Args:
            img (np.ndarray): Source image to extract object from.
            mask (np.ndarray): Binary mask indicating object location.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Cropped object image
                - np.ndarray: Cropped object mask
        """
        # Create a copy of the image with only the masked object
        obj = np.zeros_like(img)
        obj[mask] = img[mask]
        # Get the bounding box to crop the object
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Return the cropped object and its mask
        return obj[rmin:rmax+1, cmin:cmax+1], mask[rmin:rmax+1, cmin:cmax+1]

    def _random_transform(self, obj: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Applies random transformations to an object and its mask.

        Args:
            obj (np.ndarray): Object image to transform.
            mask (np.ndarray): Object mask to transform.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Transformed object
                - np.ndarray: Transformed mask
                - List[str]: List of applied transformations
        """
        transforms_applied = []
        
        if random.random() < 0.5:
            obj = np.fliplr(obj)
            mask = np.fliplr(mask)
            transforms_applied.append("flip")
        
        if random.random() < 0.5:
            obj = np.rot90(obj)
            mask = np.rot90(mask)
            transforms_applied.append("rotate")
            
        return obj, mask, transforms_applied
    
    def _find_valid_position(self, obj_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Finds a valid position to place an object within the image boundaries.

        Args:
            obj_shape (Tuple[int, int]): Shape of the object to be placed.

        Returns:
            Tuple[int, int]: Valid (y, x) position coordinates.
        """
        max_y = self.data_img.shape[0] - obj_shape[0]
        max_x = self.data_img.shape[1] - obj_shape[1]
        
        y = random.randint(0, max(0, max_y))
        x = random.randint(0, max(0, max_x))
        
        return y, x
    
    def _place_object(self, img: np.ndarray, obj: np.ndarray, obj_mask: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Places an object at the specified position in the image.

        Args:
            img (np.ndarray): Target image where object will be placed.
            obj (np.ndarray): Object to be placed.
            obj_mask (np.ndarray): Mask of the object.
            position (Tuple[int, int]): Position (y, x) where to place the object.

        Returns:
            np.ndarray: Image with placed object.
        """
        y, x = position
        h, w = obj.shape
        
        # Ensure we don't go out of bounds
        h = min(h, img.shape[0] - y)
        w = min(w, img.shape[1] - x)
        
        new_img = img.copy()
        # Create the target region mask
        target_region = np.zeros((h, w), dtype=bool)
        # Use only the valid portion of the mask
        target_region[:min(h, obj_mask.shape[0]), :min(w, obj_mask.shape[1])] = \
            obj_mask[:min(h, obj_mask.shape[0]), :min(w, obj_mask.shape[1])]
        
        # Place the object using the properly sized mask
        placement_region = new_img[y:y+h, x:x+w]
        placement_region[target_region] = obj[:h, :w][target_region]
        new_img[y:y+h, x:x+w] = placement_region
        
        return new_img
    
    def augment(self) -> Tuple[np.ndarray, np.ndarray]:
        """Performs augmentation on both data and label images.

        Randomly selects and transforms approximately one-third of the objects
        in the image pair, with a 50% chance of moving each selected object
        to a new position.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Augmented data image
                - np.ndarray: Augmented label image
        """
        num_objects_to_move = max(1, self.num_objects // 3)
        objects_to_move = random.sample(range(1, self.num_objects + 1), num_objects_to_move)
        
        augmented_data = self.data_img.copy()
        augmented_label = self.label_img.copy()
        
        for obj_label in objects_to_move:
            # Get the original mask for this object
            mask = self.labeled_img == obj_label
            
            # Extract objects using mask
            data_obj, obj_mask = self._extract_object(self.data_img, mask)
            label_obj, _ = self._extract_object(self.label_img, mask)
            
            # Transform data object and get the transformation record
            data_obj, obj_mask, transforms = self._random_transform(data_obj, obj_mask)
            
            # Apply exactly the same transformations to label object
            label_obj = label_obj.copy()  # ensure we have a copy to transform
            if "flip" in transforms:
                label_obj = np.fliplr(label_obj)
            if "rotate" in transforms:
                label_obj = np.rot90(label_obj)
            
            if random.random() < 0.5:  # 50% chance to move
                new_pos = self._find_valid_position(data_obj.shape)
                
                # Fill original position
                augmented_data[mask] = self.background_value
                augmented_label[mask] = 0
                
                # Place objects using the same mask and position
                augmented_data = self._place_object(augmented_data, data_obj, obj_mask, new_pos)
                augmented_label = self._place_object(augmented_label, label_obj, obj_mask, new_pos)
        
        return augmented_data, augmented_label

def process_folder(data_path: Path, label_path: Path, output_data_path: Path, output_label_path: Path, num_augmentations: int = 3):
    """Processes all images in the input folders to create augmented versions.

    Args:
        data_path (Path): Path to the folder containing data images.
        label_path (Path): Path to the folder containing label images.
        output_data_path (Path): Path where augmented data images will be saved.
        output_label_path (Path): Path where augmented label images will be saved.
        num_augmentations (int, optional): Number of augmented versions to create for each image. Defaults to 3.

    Raises:
        FileNotFoundError: If data_path or label_path doesn't exist.
        ValueError: If number of data and label images don't match or if filenames don't correspond.
    """
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError("Data or label path does not exist")
    
    output_data_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)

    data_files = [Path(item) for item in fb.get_image_names(str(data_path), None, 'tif')]
    label_files = [Path(item) for item in fb.get_image_names(str(label_path), None, 'tif')]

    if len(data_files) != len(label_files):
        raise ValueError("Data and label images do not match")
    for i in range(len(data_files)):
        if data_files[i].name.split('-')[:2] != label_files[i].name.split('-')[:2]:
            raise ValueError(f'File names do not match: {data_files[i].name} and {label_files[i].name}')

    for data_file, label_file in zip(data_files, label_files):
        data_img = cv2.imread(str(data_file), cv2.IMREAD_UNCHANGED)
        label_img = read_tif_image_label(str(label_file))
        
        for aug_idx in range(num_augmentations):
            augmenter = ImageAugmenter(data_img, label_img)
            augmented_data, augmented_label = augmenter.augment()
            
            # Create filenames with augmentation index
            aug_data_name = data_file.name.replace('-harmonized', f'aug{aug_idx+1}-augmented')
            aug_label_name = label_file.name.replace('-preciseLabel', f'aug{aug_idx+1}-augmented')
            
            cv2.imwrite(str(output_data_path / aug_data_name), 
                        augmented_data, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            cv2.imwrite(str(output_label_path / aug_label_name), 
                        augmented_label, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

if __name__ == '__main__':
    data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/image/')
    label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-refined/')
    output_data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/image/')
    output_label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/label/')
    
    process_folder(data_path, label_path, output_data_path, output_label_path, num_augmentations=3)
