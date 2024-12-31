import cv2
import numpy as np
import random
import sys

sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from skimage import measure
from pathlib import Path
from typing import Tuple, List
from src.workflow_tools.database.s4refine_label import read_tif_image_label
from src.workflow_tools.database.s4padding import pad_data_image, pad_label_image, get_padding_mask
from src.API_functions.Images import file_batch as fb

class ImageAugmenter:
    """A class for performing image augmentation on data and label image pairs.

    This updated version introduces higher probabilities of transformations
    and improved docstrings for clarity and maintainability.

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
        """Perform multiple random transformations with higher combination probability.

        Now includes flips, rotation, and scaling. Each transformation has an increased chance
        of being applied to achieve more variation in the augmented data.
        """
        transforms_applied = []
        
        # Generate all random numbers at once
        rands = {
            'horizontal': random.random(),
            'vertical': random.random(),
            'rotation': random.random(),
            'scale': random.random()
            # Removed 'brightness' and 'blur'
        }
        
        # Scale transformations (mutually exclusive)
        max_allowed_size = min(self.data_img.shape[0]//2, self.data_img.shape[1]//2)  # Max 1/2 of image size
        min_size = 6  # Minimum object size to allow scaling down
        
        if rands['scale'] < 0.3 and obj.shape[0]*2 <= max_allowed_size and obj.shape[1]*2 <= max_allowed_size:
            # Scale up 2x with nearest neighbor interpolation
            new_size = (obj.shape[1]*2, obj.shape[0]*2)  # Note: cv2.resize takes (width, height)
            obj = cv2.resize(obj, new_size, interpolation=cv2.INTER_LINEAR)  # Linear for data
            mask_uint8 = mask.astype(np.uint8)
            mask_uint8 = cv2.resize(mask_uint8, new_size, interpolation=cv2.INTER_NEAREST)
            mask = mask_uint8.astype(bool)
            transforms_applied.append("scale_up_2x")
            
        elif rands['scale'] < 0.6 and obj.shape[0] > min_size and obj.shape[1] > min_size:
            # Scale down 0.5x with nearest neighbor interpolation
            new_size = (obj.shape[1]//2, obj.shape[0]//2)  # Integer division for exact scaling
            obj = cv2.resize(obj, new_size, interpolation=cv2.INTER_LINEAR)  # Linear for data
            mask_uint8 = mask.astype(np.uint8)
            mask_uint8 = cv2.resize(mask_uint8, new_size, interpolation=cv2.INTER_NEAREST)
            mask = mask_uint8.astype(bool)
            transforms_applied.append("scale_down_0.5x")
        
        # Horizontal flip (50% probability)
        if rands['horizontal'] < 0.5:
            obj = np.fliplr(obj)
            mask = np.fliplr(mask)
            transforms_applied.append("horizontal_flip")
        
        # Vertical flip (50% probability)
        if rands['vertical'] < 0.5:
            obj = np.flipud(obj)
            mask = np.flipud(mask)
            transforms_applied.append("vertical_flip")
        
        # Rotation (independent probabilities for each angle)
        if rands['rotation'] < 0.25:  # 90 degrees
            obj = np.rot90(obj, k=1)
            mask = np.rot90(mask, k=1)
            transforms_applied.append("rotate_90")
        elif rands['rotation'] < 0.5:  # 180 degrees
            obj = np.rot90(obj, k=2)
            mask = np.rot90(mask, k=2)
            transforms_applied.append("rotate_180")
        elif rands['rotation'] < 0.75:  # 270 degrees
            obj = np.rot90(obj, k=3)
            mask = np.rot90(mask, k=3)
            transforms_applied.append("rotate_270")
        
        # Removed0 brightness and blur transformations
        
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
        """Optimized object placement with dimension safety checks and overlay handling"""
        y, x = position
        h, w = obj_mask.shape  # Use mask shape as reference

        # Ensure we don't go out of bounds
        h = min(h, img.shape[0] - y)
        w = min(w, img.shape[1] - x)

        # Get valid region for placement
        placement_region = img[y:y+h, x:x+w]
        valid_mask = obj_mask[:h, :w]

        # Overlay the object on top of existing data
        placement_region[valid_mask] = obj[:h, :w][valid_mask]

        return img

    def augment(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform augmentation on the pre-padded data, label, and mask."""
        objects_to_move = list(range(1, self.num_objects + 1))
        # Randomly shuffle objects to avoid positional bias
        random.shuffle(objects_to_move)
        
        # Create copies to avoid modifying the original images
        augmented_data = self.data_img.copy()
        augmented_label = self.label_img.copy()
        augmented_mask = self.mask.copy()
        
        # Track which objects have been moved
        moved_objects = set()
        target_moves = len(objects_to_move) // 2  # Try to move at least half
        
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
            
            # Apply same transformations to label and mask
            for transform in transforms:
                if "scale_up_2x" in transform:
                    new_size = (label_obj.shape[1]*2, label_obj.shape[0]*2)
                    label_obj = cv2.resize(label_obj, new_size, interpolation=cv2.INTER_NEAREST)
                    mask_obj = cv2.resize(mask_obj, new_size, interpolation=cv2.INTER_NEAREST)
                elif "scale_down_0.5x" in transform:
                    new_size = (label_obj.shape[1]//2, label_obj.shape[0]//2)
                    label_obj = cv2.resize(label_obj, new_size, interpolation=cv2.INTER_NEAREST)
                    mask_obj = cv2.resize(mask_obj, new_size, interpolation=cv2.INTER_NEAREST)
                elif "horizontal_flip" in transform:
                    label_obj = np.fliplr(label_obj)
                    mask_obj = np.fliplr(mask_obj)
                elif "vertical_flip" in transform:
                    label_obj = np.flipud(label_obj)
                    mask_obj = np.flipud(mask_obj)
                elif "rotate_90" in transform:
                    label_obj = np.rot90(label_obj, k=1)
                    mask_obj = np.rot90(mask_obj, k=1)
                elif "rotate_180" in transform:
                    label_obj = np.rot90(label_obj, k=2)
                    mask_obj = np.rot90(mask_obj, k=2)
                elif "rotate_270" in transform:
                    label_obj = np.rot90(label_obj, k=3)
                    mask_obj = np.rot90(mask_obj, k=3)
            
            # If object is bigger than the image, try scaling down by half.
            if data_obj.shape[0] > self.data_img.shape[0] or data_obj.shape[1] > self.data_img.shape[1]:
                new_size = (data_obj.shape[1] // 2, data_obj.shape[0] // 2)
                if new_size[0] > 0 and new_size[1] > 0:
                    data_obj = cv2.resize(data_obj, new_size, interpolation=cv2.INTER_LINEAR)
                    mask_uint8 = obj_mask.astype(np.uint8)
                    mask_uint8 = cv2.resize(mask_uint8, new_size, interpolation=cv2.INTER_NEAREST)
                    obj_mask = mask_uint8.astype(bool)

                    label_obj = cv2.resize(label_obj, new_size, interpolation=cv2.INTER_NEAREST)
                    mask_obj = cv2.resize(mask_obj, new_size, interpolation=cv2.INTER_NEAREST)

            # Try to move object if we haven't met our target number of moves
            should_try_move = len(moved_objects) < target_moves or random.random() < 0.3
            
            if should_try_move:
                # Try up to 5 different positions for each object
                for _ in range(5):
                    new_pos = self._find_valid_position(data_obj.shape)
                    y, x = new_pos

                    # Check if new position overlaps with any moved objects
                    target_mask = self.mask[y:y+data_obj.shape[0], x:x+data_obj.shape[1]]
                    if np.all(target_mask[obj_mask] == 1):
                        # Place the object on top
                        augmented_data = self._place_object(augmented_data, data_obj, obj_mask, new_pos)
                        augmented_label = self._place_object(augmented_label, label_obj, obj_mask, new_pos)
                        augmented_mask = self._place_object(augmented_mask, mask_obj, obj_mask, new_pos)
                        moved_objects.add(obj_label)
                        break
        
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
                        augmented_mask.astype(np.float32), [cv2.IMWRITE_TIFF_COMPRESSION, 1])

if __name__ == '__main__':
    data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/image/')
    label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-refined/')
    output_data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/image/')
    output_label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/label/')
    output_mask_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/mask/')
    process_folder(data_path, label_path, output_data_path, output_label_path, output_mask_path, num_augmentations=10)
