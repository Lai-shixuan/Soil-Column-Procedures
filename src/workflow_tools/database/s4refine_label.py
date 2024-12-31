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

def process_white_objects(img, data_img, original_filename) -> List[Dict[str, Any]]:
    """Process white objects in the image and return their properties"""
    # Label connected components
    labeled_img = measure.label(img)
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

def process_tif_folder(label_path: Path, data_path: Path, output: Path, iqr_threshold: float = 0.75):
    """Process all TIF images in the folder and create a DataFrame
    
    Args:
        label_path: Path to label images
        data_path: Path to data images
        output: Path to save refined labels
        iqr_threshold: Multiplier for IQR to determine outlier threshold (default: 0.75)
    """
    all_pores = []

    labels = fb.get_image_names(str(label_path), None, 'tif')
    datas = fb.get_image_names(str(data_path), None, 'tif')
    names = [Path(label).stem for label in labels]

    for label_file, data_file, name in zip(labels, datas, names):
        label_img = read_tif_image_label(label_file)
        data_img = cv2.imread(str(data_file), cv2.IMREAD_UNCHANGED)

        objects = process_white_objects(label_img, data_img, name)
        all_pores.extend(objects)
    
    df = pd.DataFrame(all_pores)
    df = df[['id', 'original_image', 'size', 'avg_value']]
    
    # Use IQR method to detect outliers
    Q1 = df['avg_value'].quantile(0.25)
    Q3 = df['avg_value'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + iqr_threshold * IQR  # Using configurable IQR threshold
    
    # Mark outliers (True for outliers, False for valid data)
    df['outlier'] = df['avg_value'] > upper_bound
    
    # Process and save modified label images
    for name in df['original_image'].unique():
        label_file = next(label_path.glob(f"{name}*.tif"))
        label_img = read_tif_image_label(label_file)
        labeled_img = measure.label(label_img)
        
        # Get outlier IDs for this image
        outlier_ids = df[(df['original_image'] == name) & (df['outlier'])]['id'].values
        
        # Create new image with outliers removed
        new_img = label_img.copy().astype(np.float32)  # ensure float32 type
        for idx in outlier_ids:
            mask = labeled_img == (idx + 1)  # regionprops index starts from 1
            new_img[mask] = 0
            
        # Save modified image as 32-bit float TIF
        cv2.imwrite(str(output_path / f"{name}.tif"), new_img, 
                    [cv2.IMWRITE_TIFF_COMPRESSION, 1])  # uncompressed TIFF
    
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
    input_gray_data_path = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/3.Harmonized/image')
    output_path = Path(r'/mnt/g/DL_Data_raw/version8-low-precise/4.Converted/label-refined')

    if not input_label_path.exists():
        raise ValueError(f"Folder not found: {input_label_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Configure IQR threshold
    IQR_THRESHOLD = 0.75
    result_df = process_tif_folder(input_label_path, input_gray_data_path, output_path, IQR_THRESHOLD)
    result_df.to_csv(Path("output.csv"), index=False, lineterminator='\n')

    # Calculate upper bound for plotting
    Q1 = result_df['avg_value'].quantile(0.25)
    Q3 = result_df['avg_value'].quantile(0.75)
    upper_bound = Q3 + IQR_THRESHOLD * (Q3 - Q1)

    # Create visualization
    fig = plot_analysis(result_df, upper_bound)
    plt.show()
