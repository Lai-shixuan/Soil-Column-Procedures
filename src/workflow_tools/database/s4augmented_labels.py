import cv2
import numpy as np
import random
import sys

sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from skimage import measure
from pathlib import Path
from typing import Tuple, List, Optional
from src.workflow_tools.database.s4refine_label import read_tif_image_label
from src.workflow_tools.database.s4padding import pad_data_image, pad_label_image, get_padding_mask
from src.API_functions.Images import file_batch as fb

class ImageObject:
    """Base class for image objects"""
    def __init__(self, object_id: int, full_image: np.ndarray, object_mask: np.ndarray):
        self.object_id = object_id
        self.full_image = full_image
        
        # Get object indices within its bounding box
        rows, cols = np.nonzero(object_mask)
        if len(rows) == 0:
            raise ValueError(f"Empty object mask for object {object_id}")
            
        # Store original position
        self.orig_bbox = {
            'rmin': rows.min(),
            'rmax': rows.max() + 1,
            'cmin': cols.min(),
            'cmax': cols.max() + 1
        }
        
        # Extract object and its mask
        self.obj_data = self.full_image[
            self.orig_bbox['rmin']:self.orig_bbox['rmax'],
            self.orig_bbox['cmin']:self.orig_bbox['cmax']
        ].copy()
        
        self.obj_mask = object_mask[
            self.orig_bbox['rmin']:self.orig_bbox['rmax'],
            self.orig_bbox['cmin']:self.orig_bbox['cmax']
        ].copy()
        
        # Current position (will be updated during placement)
        self.current_bbox = self.orig_bbox.copy()
        self.transforms = []
        
    def get_size(self) -> Tuple[int, int]:
        """Get current object size"""
        return self.obj_data.shape

    def _apply_transform(self, transform: str) -> None:
        """Apply a single transformation using match-case"""
        match transform:
            case "scale_up_2x":
                new_size = (self.obj_data.shape[1]*2, self.obj_data.shape[0]*2)
                # Check if scaled up size would exceed full image dimensions
                if (new_size[1] > self.full_image.shape[0] or 
                    new_size[0] > self.full_image.shape[1]):
                    # Fallback to scale down if too large
                    new_size = (self.obj_data.shape[1]//2, self.obj_data.shape[0]//2)
                self.obj_data = cv2.resize(self.obj_data, new_size, interpolation=cv2.INTER_LINEAR)
                self.obj_mask = cv2.resize(self.obj_mask.astype(np.float32), new_size, 
                                        interpolation=cv2.INTER_NEAREST) > 0.5
                
            case "scale_down_0.5x":
                new_size = (self.obj_data.shape[1]//2, self.obj_data.shape[0]//2)
                self.obj_data = cv2.resize(self.obj_data, new_size, interpolation=cv2.INTER_LINEAR)
                self.obj_mask = cv2.resize(self.obj_mask.astype(np.float32), new_size, 
                                        interpolation=cv2.INTER_NEAREST) > 0.5
                
            case "horizontal_flip":
                self.obj_data = np.fliplr(self.obj_data)
                self.obj_mask = np.fliplr(self.obj_mask)
                
            case "vertical_flip":
                self.obj_data = np.flipud(self.obj_data)
                self.obj_mask = np.flipud(self.obj_mask)
                
            case _ if transform.startswith("rotate_"):
                k = int(transform.split('_')[1]) // 90
                self.obj_data = np.rot90(self.obj_data, k=k)
                self.obj_mask = np.rot90(self.obj_mask, k=k)

    def apply_transforms(self, transforms: List[str]) -> None:
        """Apply a sequence of transformations"""
        self.transforms = transforms
        for transform in transforms:
            self._apply_transform(transform)

    def update_position(self, new_y: int, new_x: int) -> None:
        """Update object's position in the full image"""
        self.current_bbox = {
            'rmin': new_y,
            'rmax': new_y + self.obj_data.shape[0],
            'cmin': new_x,
            'cmax': new_x + self.obj_data.shape[1]
        }

    def place_in_image(self, target_img: np.ndarray, position: Tuple[int, int]) -> None:
        """Place object in target image at specified position"""
        y, x = position
        h, w = self.get_size()
        
        # Ensure we don't go out of bounds
        h = min(h, target_img.shape[0] - y)
        w = min(w, target_img.shape[1] - x)
        
        mask = self.obj_mask[:h, :w]
        target_img[y:y+h, x:x+w][mask] = self.obj_data[:h, :w][mask]
        self.update_position(y, x)

class DataObject(ImageObject):
    """Class for data image objects"""
    def find_valid_position(self, boundary: np.ndarray, max_attempts: int = 5) -> Optional[Tuple[int, int]]:
        """Find a valid position within boundary constraints"""
        h, w = self.get_size()
        max_y = boundary.shape[0] - h
        max_x = boundary.shape[1] - w
        
        for _ in range(max_attempts):
            y = random.randint(0, max(0, max_y))
            x = random.randint(0, max(0, max_x))
            
            target_region = boundary[y:y+h, x:x+w]
            if np.all(target_region[self.obj_mask] == 1):
                return y, x
        return None

class LabelObject(ImageObject):
    """Class for label image objects"""
    def copy_transforms_from_data(self, data_obj: DataObject) -> None:
        """Copy transforms from corresponding data object"""
        self.apply_transforms(data_obj.transforms)
        self.update_position(data_obj.current_bbox['rmin'], 
                           data_obj.current_bbox['cmin'])

class AdditionalObject(ImageObject):
    """Class for additional mask objects"""
    def copy_transforms_from_data(self, data_obj: DataObject) -> None:
        """Copy transforms from corresponding data object"""
        self.apply_transforms(data_obj.transforms)
        self.update_position(data_obj.current_bbox['rmin'], 
                           data_obj.current_bbox['cmin'])

class ImageAugmenter:
    """Updated augmenter using object-oriented approach"""
    def __init__(self, data_img: np.ndarray, label_img: np.ndarray, additional_img: np.ndarray = None, mask: np.ndarray = None):
        self.data_img = data_img
        self.label_img = label_img
        self.additional_img = additional_img
        self.boundary = mask if mask is not None else np.ones_like(label_img)
        
        # Split into objects
        self.splited_img, self.num_objects = measure.label(self.label_img, return_num=True)
        self.data_objects: List[DataObject] = []
        self.label_objects: List[LabelObject] = []
        self.additional_objects: List[AdditionalObject] = []
        
        # Create objects
        for obj_id in range(1, self.num_objects + 1):
            object_mask = (self.splited_img == obj_id)
            
            data_obj = DataObject(obj_id, self.data_img, object_mask)
            self.data_objects.append(data_obj)
            
            label_obj = LabelObject(obj_id, self.label_img, object_mask)
            self.label_objects.append(label_obj)
            
            if additional_img is not None:
                additional_obj = AdditionalObject(obj_id, self.additional_img, object_mask)
                self.additional_objects.append(additional_obj)

    def _generate_random_transforms(self) -> List[str]:
        """Generate random transformation sequence"""
        transforms = []
        
        # Scale transformations
        scale_rand = random.random()
        if scale_rand < 0.3:
            transforms.append("scale_up_2x")
        elif scale_rand < 0.6:
            transforms.append("scale_down_0.5x")
            
        # Flips
        if random.random() < 0.5:
            transforms.append("horizontal_flip")
        if random.random() < 0.5:
            transforms.append("vertical_flip")
            
        # Rotation
        rot_rand = random.random()
        if rot_rand < 0.25:
            transforms.append("rotate_90")
        elif rot_rand < 0.5:
            transforms.append("rotate_180")
        elif rot_rand < 0.75:
            transforms.append("rotate_270")
            
        return transforms

    def augment(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Perform augmentation using object-oriented approach"""
        augmented_data = self.data_img.copy()
        augmented_label = self.label_img.copy()
        augmented_additional = self.additional_img.copy() if self.additional_img is not None else None
        
        # Randomly shuffle objects
        indices = list(range(len(self.data_objects)))
        random.shuffle(indices)
        
        for idx in indices:
            data_obj = self.data_objects[idx]
            label_obj = self.label_objects[idx]
            
            # Generate and apply transforms for data object
            transforms = self._generate_random_transforms()
            data_obj.apply_transforms(transforms)
            
            # Find valid position
            position = data_obj.find_valid_position(self.boundary)
            if position is not None:
                # Place data object
                data_obj.place_in_image(augmented_data, position)
                
                # Update and place label object
                label_obj.copy_transforms_from_data(data_obj)
                label_obj.place_in_image(augmented_label, position)
                
                # Handle additional object if it exists
                if self.additional_img is not None and idx < len(self.additional_objects):
                    additional_obj = self.additional_objects[idx]
                    additional_obj.copy_transforms_from_data(data_obj)
                    additional_obj.place_in_image(augmented_additional, position)
        
        return augmented_data, augmented_label, augmented_additional

def generate_random_grid_mask(shape: Tuple[int, int], 
                            grid_size: Tuple[int, int] = (8, 8),
                            probability: float = 0.5) -> np.ndarray:
    """Generate a random grid mask with rectangles"""
    mask = np.zeros(shape, dtype=np.float32)
    height, width = shape
    rows, cols = grid_size
    
    cell_height = height // rows
    cell_width = width // cols
    
    for i in range(rows):
        for j in range(cols):
            if random.random() < probability:
                y_start = i * cell_height
                x_start = j * cell_width
                y_end = (i + 1) * cell_height
                x_end = (j + 1) * cell_width
                mask[y_start:y_end, x_start:x_end] = 1.0
                
    return mask

def process_folder(data_path: Path, label_path: Path, output_data_path: Path, 
                  output_label_path: Path, output_mask_path: Path, 
                  output_additional_path: Path, num_augmentations: int = 3):
    """Process folders with padding and mask handling."""
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError("Data or label path does not exist")
    
    output_data_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)
    output_mask_path.mkdir(parents=True, exist_ok=True)
    output_additional_path.mkdir(parents=True, exist_ok=True)

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
            # Generate random grid mask
            additional_mask = generate_random_grid_mask(padded_data.shape[:2])
            
            # Create augmenter with additional mask
            augmenter = ImageAugmenter(padded_data, padded_label, additional_mask, mask)
            augmented_data, augmented_label, augmented_additional = augmenter.augment()
            
            # Create filenames with augmentation index
            aug_data_name = data_file.name.replace('-harmonized', f'aug{aug_idx+1}-augmented')
            aug_label_name = label_file.name.replace('-preciseLabel', f'aug{aug_idx+1}-augmented')
            aug_additional_name = data_file.name.replace('-harmonized', f'aug{aug_idx+1}-additional')
            
            # Save all images
            cv2.imwrite(str(output_data_path / aug_data_name), 
                       augmented_data, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            cv2.imwrite(str(output_label_path / aug_label_name), 
                       augmented_label, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            cv2.imwrite(str(output_additional_path / aug_additional_name),
                       augmented_additional.astype(np.float32), [cv2.IMWRITE_TIFF_COMPRESSION, 1])

if __name__ == '__main__':
    data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/image/')
    label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-refined/')
    output_data_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/image/')
    output_label_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/label/')
    output_mask_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/mask/')
    output_additional_path = Path('/mnt/g/DL_Data_raw/version8-low-precise/5.1.Augmented/additional/')
    
    process_folder(data_path, label_path, output_data_path, output_label_path, 
                  output_mask_path, output_additional_path, num_augmentations=10)
