# TODO, check the density of the histogram, maybe wrong

import cv2
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from skimage import measure
from pathlib import Path
from scipy.stats import norm
from typing import List, Dict, Any
from src.API_functions.Images import file_batch as fb

def read_tif_image_label(file_path):
    """Read a TIF image using OpenCV and ensure it's binary (0 and 1)"""
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to read image: {file_path}")
    if not np.array_equal(np.unique(img), np.array([0., 1.])):
        raise ValueError(f"Image {file_path} is not binary (0 and 1)")

    return img

def process_white_objects(label_img, data_img, original_filename) -> List[Dict[str, Any]]:
    """Process white objects in the image and return their properties"""
    # Label connected components
    labeled_img = measure.label(label_img)
    properties = measure.regionprops(labeled_img)
    
    object_data = []
    for idx, prop in enumerate(properties):
        # Get the mask for current pore
        mask = labeled_img == prop.label
        # Calculate average pixel value in data_img for the current pore
        avg_value = np.mean(data_img[mask])
        
        object_data.append({
            'id': idx,
            'original_image': original_filename,
            'size': prop.area,
            'avg_value': avg_value
        })
    
    return object_data

def process_tif_folder(label_path: Path, data_path: Path, output: Path, debug_output: Path, 
                      iqr_threshold: float = 0.75, use_global_threshold: bool = False):
    """Process all TIF images in the folder and create a DataFrame
    
    Args:
        label_path: Path to label images
        data_path: Path to data images
        output: Path to save refined labels
        debug_output: Path to save marked debug images
        iqr_threshold: Multiplier for IQR to determine outlier threshold (default: 0.75)
        use_global_threshold: If True, use global statistics for thresholding (default: False)
    """
    all_pores = []
    image_data = {}

    labels = fb.get_image_names(str(label_path), None, 'tif')
    datas = fb.get_image_names(str(data_path), None, 'png')
    names = [Path(label).stem for label in labels]

    for label_file, data_file, name in zip(labels, datas, names):
        label_img = read_tif_image_label(label_file)
        data_img = cv2.imread(str(data_file), cv2.IMREAD_UNCHANGED)
        
        # Store the images
        image_data[name] = {
            'label': label_img,
            'data': data_img
        }

        objects = process_white_objects(label_img, data_img, name)
        all_pores.extend(objects)
    
    df = pd.DataFrame(all_pores)
    df = df[['id', 'original_image', 'size', 'avg_value']]

    # Calculate global thresholds if needed
    if use_global_threshold:
        Q1 = df['avg_value'].quantile(0.25)
        Q3 = df['avg_value'].quantile(0.75)
        IQR = Q3 - Q1
        global_upper_bound = Q3 + iqr_threshold * IQR
        global_lower_bound = Q1 - iqr_threshold * IQR
    
    # Process each image separately
    for name in df['original_image'].unique():
        img_df = df[df['original_image'] == name].copy()
        
        # Use either global or image-specific thresholds
        if use_global_threshold:
            img_upper_bound = global_upper_bound
            img_lower_bound = global_lower_bound
        else:
            Q1 = img_df['avg_value'].quantile(0.25)
            Q3 = img_df['avg_value'].quantile(0.75)
            IQR = Q3 - Q1
            img_upper_bound = Q3 + iqr_threshold * IQR
            img_lower_bound = Q1 - iqr_threshold * IQR
        
        # Mark outliers for this image using .loc
        img_df.loc[:, 'outlier'] = img_df['avg_value'] > img_upper_bound
        outlier_ids = img_df[img_df['outlier']]['id'].values
        
        # Use stored images
        label_img = image_data[name]['label']
        data_img = image_data[name]['data']
        labeled_img = measure.label(label_img)
        
        # Create output images
        refined_img = label_img.copy().astype(np.float32)
        debug_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
        
        # Mark valid labels as white
        debug_img[label_img == 1] = [255, 255, 255]
        
        # Mark outliers (high values with labels) as red
        for idx in outlier_ids:
            mask = labeled_img == (idx + 1)
            refined_img[mask] = 0
            debug_img[mask] = [0, 0, 255]  # BGR format: Red is [0, 0, 255]
        
        # Mark missing parts (low values without labels) as blue
        missing_mask = (data_img < img_lower_bound) & (label_img == 0)
        refined_img[missing_mask] = 1  # Mark missing parts in refined_img
        debug_img[missing_mask] = [255, 0, 0]  # BGR format: Blue is [255, 0, 0]
        
        # Save images
        cv2.imwrite(str(output / f"{name}.tif"), refined_img, 
                    [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        cv2.imwrite(str(debug_output / f"{name}.png"), debug_img)
        
        # Update main dataframe with outlier information using .loc
        df.loc[img_df.index, 'outlier'] = img_df['outlier']
    
    return df

def plot_analysis(df: pd.DataFrame, upper_bound: float):
    """Create combined figure with scatter plot and histogram"""
    fig = plt.figure(figsize=(15, 6))
    
    # Scatter plot (left subplot)
    plt.subplot(121)
    outlier_mask = df['avg_value'] > upper_bound
    valid_data = df[~outlier_mask]
    outlier_data = df[outlier_mask]
    
    plt.scatter(valid_data['size'], valid_data['avg_value'], 
                alpha=0.5, c='blue', label='Valid')
    plt.scatter(outlier_data['size'], outlier_data['avg_value'], 
                alpha=0.5, c='red', label='Outlier')
    
    plt.xscale('log')
    plt.xlabel('Pore Area (pixels)')
    plt.ylabel('Average Gray Value')
    plt.title('Pore Size vs Average Gray Value')
    plt.grid(True)
    plt.legend()
    
    # Histogram plot (right subplot)
    plt.subplot(122)
    x_min = df['avg_value'].min() * 0.99
    x_max = df['avg_value'].max() * 1.01
    
    plt.hist(df['avg_value'], bins=50, density=True, alpha=0.7, edgecolor='black')
    
    # Fit normal distribution
    mu, std = norm.fit(df['avg_value'])
    bin_edges = np.linspace(x_min, x_max, 100)
    pdf = norm.pdf(bin_edges, mu, std)
    
    plt.plot(bin_edges, pdf, 'g-', lw=2, label=f'Normal Fit (μ={mu:.3f}, σ={std:.3f})')
    plt.axvline(upper_bound, color='red', linestyle=':', 
                label=f'IQR bound ({upper_bound:.3f})')
    
    plt.xlim(x_min, x_max)
    plt.xlabel('Average Gray Value')
    plt.ylabel('Density')
    plt.title('Distribution of Average Gray Values with Normal Fit')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    input_label_path = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-origin')
    input_gray_data_path = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/8bit')
    output_path = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-refined')
    debug_output_path = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-debug')

    if not input_label_path.exists():
        raise ValueError(f"Folder not found: {input_label_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    if not debug_output_path.exists():
        debug_output_path.mkdir(parents=True, exist_ok=True)

    # Configure parameters
    IQR_THRESHOLD = 1 
    USE_GLOBAL_THRESHOLD = True  # Set to True to use global statistics

    result_df = process_tif_folder(input_label_path, input_gray_data_path, 
                                output_path, debug_output_path, 
                                IQR_THRESHOLD, USE_GLOBAL_THRESHOLD)
    result_df.to_csv(Path("output.csv"), index=False, lineterminator='\n')

    # Always use global threshold for visualization
    Q1 = result_df['avg_value'].quantile(0.25)
    Q3 = result_df['avg_value'].quantile(0.75)
    global_upper_bound = Q3 + IQR_THRESHOLD * (Q3 - Q1)

    # Create visualization with global threshold
    fig = plot_analysis(result_df, global_upper_bound)
    plt.show()
