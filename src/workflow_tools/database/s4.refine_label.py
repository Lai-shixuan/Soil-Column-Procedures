# TODO, make the IQR threshold configurable
# TODO, make 2 figures into 2 functions
# TODO, check the density of the histogram, maybe wrong

import cv2
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm  # Add to import section

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from skimage import measure
from pathlib import Path
from src.API_functions.Images import file_batch as fb

def read_tif_image(file_path):
    """Read a TIF image using OpenCV and ensure it's binary (0 and 1)"""
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to read image: {file_path}")
    if not np.array_equal(np.unique(img), np.array([0., 1.])):
        raise ValueError(f"Image {file_path} is not binary (0 and 1)")

    return img

def process_white_objects(img, data_img, original_filename):
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

def process_tif_folder(label_path: Path, data_path: Path, output: Path):
    """Process all TIF images in the folder and create a DataFrame"""
    all_pores = []

    labels = fb.get_image_names(str(label_path), None, 'tif')
    datas = fb.get_image_names(str(data_path), None, 'tif')
    names = [Path(label).stem for label in labels]

    for label_file, data_file, name in zip(labels, datas, names):
        label_img = read_tif_image(label_file)
        data_img = cv2.imread(str(data_file), cv2.IMREAD_UNCHANGED)

        objects = process_white_objects(label_img, data_img, name)
        all_pores.extend(objects)
    
    df = pd.DataFrame(all_pores)
    df = df[['id', 'original_image', 'size', 'avg_value']]
    
    # Use IQR method to detect outliers
    Q1 = df['avg_value'].quantile(0.25)
    Q3 = df['avg_value'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 0.75 * IQR  # Using 0.5 times IQR as threshold
    
    # Mark outliers (True for outliers, False for valid data)
    df['outlier'] = df['avg_value'] > upper_bound
    
    # Process and save modified label images
    for name in df['original_image'].unique():
        label_file = next(label_path.glob(f"{name}*.tif"))
        label_img = read_tif_image(label_file)
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

if __name__ == '__main__':
    input_label_path = Path(r'g:\DL_Data_raw\version8-low-precise\4.Converted\label-origin')
    input_gray_data_path = Path(r'g:\DL_Data_raw\version8-low-precise\3.Harmonized\image')
    output_path = Path(r'g:\DL_Data_raw\version8-low-precise\4.Converted\label-refined')

    if not input_label_path.exists():
        raise ValueError(f"Folder not found: {input_label_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    result_df = process_tif_folder(input_label_path, input_gray_data_path, output_path)
    result_df.to_csv(Path("output.csv"), index=False, lineterminator='\n')

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Add IQR boundary
    Q1 = result_df['avg_value'].quantile(0.25)
    Q3 = result_df['avg_value'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 0.75 * IQR
    ax2.axvline(upper_bound, color='red', linestyle=':', 
                label=f'IQR bound ({upper_bound:.3f})')

    # First plot: scatter plot with log scale
    outlier_mask = result_df['avg_value'] > upper_bound
    valid_data = result_df[~outlier_mask]
    outlier_data = result_df[outlier_mask]
    
    # Plot valid points in blue
    ax1.scatter(valid_data['size'], valid_data['avg_value'], 
                alpha=0.5, c='blue', label='Valid')
    # Plot outliers in red
    ax1.scatter(outlier_data['size'], outlier_data['avg_value'], 
                alpha=0.5, c='red', label='Outlier')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Pore Area (pixels)')
    ax1.set_ylabel('Average Gray Value')
    ax1.set_title('Pore Size vs Average Gray Value')
    ax1.grid(True)
    ax1.legend()

    # Second plot: histogram with IQR and normal distribution fit
    x_min = result_df['avg_value'].min() * 0.99
    x_max = result_df['avg_value'].max() * 1.01
    
    # Histogram
    hist, bin_edges = np.histogram(result_df['avg_value'], bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.hist(result_df['avg_value'], bins=50, density=True, alpha=0.7, edgecolor='black')
    
    # Fit normal distribution
    mu, std = norm.fit(result_df['avg_value'])
    pdf = norm.pdf(bin_centers, mu, std)
    
    # Plot the fitted curve
    ax2.plot(bin_centers, pdf, 'g-', lw=2, label=f'Normal Fit (μ={mu:.3f}, σ={std:.3f})')
    
    ax2.set_xlim(x_min, x_max)
    ax2.set_xlabel('Average Gray Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Average Gray Values with Normal Fit')
    
    # Update legend to show only normal fit and IQR bound
    ax2.legend(loc='upper left')
    
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
