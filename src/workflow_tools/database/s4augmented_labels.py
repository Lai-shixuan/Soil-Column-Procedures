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

def pad_data_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Pad data image to target size with mean value."""
    h, w = image.shape
    padded = np.zeros(target_size, dtype=image.dtype)
    
    # Calculate padding
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    
    # Copy image to padded array
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    # Fill padding with mean
    image_mean = np.mean(image)
    mask = np.zeros(target_size, dtype=np.float32)
    mask[pad_h:pad_h+h, pad_w:pad_w+w] = 1.0
    padded[mask == 0] = image_mean
    
    return padded

def pad_label_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Pad label image to target size with zeros."""
    h, w = image.shape
    padded = np.zeros(target_size, dtype=image.dtype)
    
    # Calculate padding
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    
    # Copy image to padded array
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    return padded

def get_padding_mask(image_shape: Tuple[int, int], target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Create mask indicating original image area."""
    h, w = image_shape
    mask = np.zeros(target_size, dtype=np.float32)
    
    # Calculate padding
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    
    # Mark original image area
    mask[pad_h:pad_h+h, pad_w:pad_w+w] = 1.0
    
    return mask

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

    def __init__(self, data_img: np.ndarray, label_img: np.ndarray, mask: np.ndarray = None):
        """Initialize augmenter with data, label images and optional mask"""
        self.data_img = data_img
        self.label_img = label_img
        self.mask = mask if mask is not None else np.ones_like(label_img)
        self.labeled_img = measure.label(label_img)
        self.num_objects = np.max(self.labeled_img)
        # Pre-compute background mask and value
        self.background_mask = (label_img == 0)
        self.background_value = np.mean(data_img[self.background_mask])
        
    def _extract_object(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Optimized object extraction"""
        # Get indices directly instead of using np.any
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return None, None
        
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        
        cropped_img = img[rmin:rmax+1, cmin:cmax+1].copy()
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1].copy()
        
        return cropped_img, cropped_mask

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
        """Optimized object placement"""
        y, x = position
        h, w = obj.shape
        
        # Ensure we don't go out of bounds
        h = min(h, img.shape[0] - y)
        w = min(w, img.shape[1] - x)
        
        # Create view instead of copy when possible
        placement_region = img[y:y+h, x:x+w]
        valid_mask = obj_mask[:h, :w]
        placement_region[valid_mask] = obj[:h, :w][valid_mask]
        
        return img
    
    def augment(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optimized augmentation process"""
        num_objects_to_move = max(1, self.num_objects // 3)
        objects_to_move = random.sample(range(1, self.num_objects + 1), num_objects_to_move)
        
        # Create views instead of copies where possible
        augmented_data = self.data_img.copy()
        augmented_label = self.label_img.copy()
        augmented_mask = self.mask.copy()
        
        for obj_label in objects_to_move:
            # Combine mask operations
            mask = (self.labeled_img == obj_label) & (self.mask == 1)
            
            if not np.any(mask):
                continue
                
            data_obj, obj_mask = self._extract_object(self.data_img, mask)
            if data_obj is None:
                continue
                
            label_obj, _ = self._extract_object(self.label_img, mask)
            mask_obj, _ = self._extract_object(self.mask, mask)
            
            data_obj, obj_mask, transforms = self._random_transform(data_obj, obj_mask)
            label_obj = label_obj.copy()
            mask_obj = mask_obj.copy()
            
            if "flip" in transforms:
                label_obj = np.fliplr(label_obj)
                mask_obj = np.fliplr(mask_obj)
            if "rotate" in transforms:
                label_obj = np.rot90(label_obj)
                mask_obj = np.rot90(mask_obj)
            
            if random.random() < 0.5:
                new_pos = self._find_valid_position(data_obj.shape)
                
                # Only move if new position is within original image area
                target_mask = np.zeros_like(self.mask)
                y, x = new_pos
                h, w = obj_mask.shape
                target_region = target_mask[y:y+h, x:x+w]
                if np.all(self.mask[y:y+h, x:x+w][obj_mask] == 1):
                    augmented_data[mask] = self.background_value
                    augmented_label[mask] = 0
                    
                    augmented_data = self._place_object(augmented_data, data_obj, obj_mask, new_pos)
                    augmented_label = self._place_object(augmented_label, label_obj, obj_mask, new_pos)
                    augmented_mask = self._place_object(augmented_mask, mask_obj, obj_mask, new_pos)
        
        return augmented_data, augmented_label, augmented_mask

def process_folder(data_path: Path, label_path: Path, output_data_path: Path, 
                  output_label_path: Path, output_mask_path: Path, num_augmentations: int = 3):
    """Process folders with padding and mask handling."""
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError("Data or label path does not exist")
    
    output_data_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)
    output_mask_path.mkdir(parents=True, exist_ok=True)

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
        
        # Pad images to 512x512 with different padding strategies
        padded_data = pad_data_image(data_img)
        padded_label = pad_label_image(label_img)
        mask = get_padding_mask(data_img.shape)
        
        for aug_idx in range(num_augmentations):
            augmenter = ImageAugmenter(padded_data, padded_label, mask)
            augmented_data, augmented_label, augmented_mask = augmenter.augment()
            
            # Create filenames with augmentation index
            aug_data_name = data_file.name.replace('-harmonized', f'aug{aug_idx+1}-augmented')
            aug_label_name = label_file.name.replace('-preciseLabel', f'aug{aug_idx+1}-augmented')
            aug_mask_name = data_file.name.replace('-harmonized', f'aug{aug_idx+1}-mask')
            
            cv2.imwrite(str(output_data_path / aug_data_name), 
                       augmented_data, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            cv2.imwrite(str(output_label_path / aug_label_name), 
                       augmented_label, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            # Save mask as 32-bit float TIFF
            cv2.imwrite(str(output_mask_path / aug_mask_name), 
                       augmented_mask.astype(np.float32), 
                       [cv2.IMWRITE_TIFF_COMPRESSION, 1])

if __name__ == '__main__':
    data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/image/')
    label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-refined/')
    output_data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/image/')
    output_label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/label/')
    output_mask_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/mask/')
    
    process_folder(data_path, label_path, output_data_path, output_label_path, output_mask_path, num_augmentations=3)
