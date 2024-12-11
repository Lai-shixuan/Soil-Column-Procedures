import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def process_image_folders(dataset_path, label_path, mask_path):
    # Convert string paths to Path objects
    dataset_path = Path(dataset_path)
    label_path = Path(label_path)
    mask_path = Path(mask_path)
    
    # Get all files from folders
    dataset_files = sorted([f.name for f in dataset_path.glob('*')])
    label_files = sorted([f.name for f in label_path.glob('*')])
    mask_files = sorted([f.name for f in mask_path.glob('*')])
    
    dataset_images = []
    label_images = []
    
    # Group files by their first 2 characters
    file_groups = {}
    for df in dataset_files:
        prefix = df[:2]
        if prefix not in file_groups:
            file_groups[prefix] = {'dataset': None, 'label': None, 'mask': None}
        file_groups[prefix]['dataset'] = df
        
    for lf in label_files:
        prefix = lf[:2]
        if prefix in file_groups:
            file_groups[prefix]['label'] = lf
            
    for mf in mask_files:
        prefix = mf[:2]
        if prefix in file_groups:
            file_groups[prefix]['mask'] = mf
    
    # Process matching files
    for prefix, files in file_groups.items():
        if all(files.values()):  # Only process if we have all three matching files
            # Read TIF images using PIL and convert to OpenCV format
            dataset_img = np.array(Image.open(dataset_path / files['dataset']))
            label_img = np.array(Image.open(label_path / files['label']))
            mask_img = np.array(Image.open(mask_path / files['mask']))
            
            # Extract green channel and convert to grayscale
            green_channel = dataset_img[:, :, 1]
            
            # Erode mask by 3 pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
            eroded_mask = cv2.erode(mask_img, kernel, iterations=1)
            
            # Apply eroded mask to both dataset and label
            masked_img = green_channel.copy()
            masked_img[eroded_mask == 0] = 255  # Set pixels outside mask to 255
            
            masked_label = label_img.copy()
            masked_label[eroded_mask == 0] = 0   # Set pixels outside mask to 0
            
            dataset_images.append(masked_img)
            label_images.append(masked_label)
    
    return np.array(dataset_images), np.array(label_images)

if __name__ == "__main__":
    # Configuration with Path objects
    base_path = Path('f:/3.Experimental_Data/Core_datasets/Batches/1.training-origin/')
    output_path = Path('f:/3.Experimental_Data/Core_datasets/Batches/2.training-processed')

    config = {
        'dataset_path': base_path / 'images',
        'label_path': base_path / '1st_manual',
        'mask_path': base_path / 'mask',
        'sample_idx': 5     # Index of image to display
    }
    
    # Process images
    dataset_images, label_images = process_image_folders(
        config['dataset_path'],
        config['label_path'],
        config['mask_path']
    )
    
    for i, (data, label) in enumerate(zip(dataset_images, label_images)):
        data_path = output_path / 'image' / f'{i:02d}-processed.tif'
        label_path = output_path / 'label' / f'{i:02d}-processed.tif'
        cv2.imwrite(str(data_path), data)
        cv2.imwrite(str(label_path), label)