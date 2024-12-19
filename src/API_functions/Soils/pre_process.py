import cv2
import numpy as np
import sys
import pywt
import bm3d  # Add this import at the top with other imports

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from src.API_functions.Images import file_batch as fb


def origin(img):
    return img


def adjust_gamma(img, gamma_value=1):
    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # 应用伽马校正
    return cv2.LUT(img, table)


def median(img, kernel_size: int=3):
    '''Can be used for both uint8 and float32 images'''
    return cv2.medianBlur(img, kernel_size)


def laplacian_edge_enhancement(image, alpha=1):
    """
    Enhance edges of the input image using the Laplacian operator.
    
    Parameters:
        image (numpy.ndarray): Input image object (OpenCV format, grayscale).
        alpha (float): Edge enhancement intensity coefficient, default is 1.0. Higher alpha values increase the edge enhancement effect.
        
    Returns:
        enhanced_image (numpy.ndarray): Image object after edge enhancement.
    """
    # Detect edges using the Laplacian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Convert the result to an 8-bit image for easier weighting
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # Overlay the edge information from the Laplacian onto the original image
    enhanced_image = cv2.addWeighted(image, 1, laplacian, alpha, 0)
    
    return enhanced_image


def map_range_to_255(image, range_min, range_max):
    """
    Map a specific range of the original image to 0-255 and reduce the other parts of the image.
    
    Parameters:
        image (numpy.ndarray): Input image object (OpenCV format, grayscale).
        range_min (int): Minimum value of the range to be mapped.
        range_max (int): Maximum value of the range to be mapped.
        
    Returns:
        mapped_image (numpy.ndarray): Image object after mapping the specified range.
    """
    # Create a copy of the image to modify
    mapped_image = image.copy()
    
    # Create a mask for the specified range
    mask = cv2.inRange(image, range_min, range_max)
    
    # Apply the mask to the image
    mapped_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Normalize the masked image to the range 0-255
    mapped_image = cv2.normalize(mapped_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Set pixels below range_min to 0
    mapped_image[image < range_min] = 0
    
    # Set pixels above range_max to 255
    mapped_image[image > range_max] = 255
    
    return mapped_image


def high_boost_filter(image, sigma=1.0, k=1.5):
    """
    Enhance small, delicate edges (like blood vessels) in a grayscale image using high-boost filtering.

    Parameters:
        image (numpy.ndarray): Input grayscale image with intensity values ranging from 0 to 65535.
        sigma (float): Standard deviation for Gaussian blur, controlling the smoothness level.
        k (float): Boost factor for enhancing high-frequency components. Higher values increase edge enhancement.

    Returns:
        enhanced_image (numpy.ndarray): Image with enhanced edges and details.
    """
    # Convert image to float32 to prevent overflow and maintain precision
    image_float = image.astype(np.float32)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(image_float, (0, 0), sigma)

    # Correct implementation of high-boost filtering
    # enhanced_image = image + k * (image - blurred)
    enhanced_image = cv2.addWeighted(image_float, 1 + k, blurred, -k, 0)

    # Clip values to the original data range and convert back to the original data type
    enhanced_image = np.clip(enhanced_image, 0, 65535).astype(image.dtype)

    return enhanced_image


def map_range_to_1(image, range_min, range_max):
    """
    Map a specific range of the original image to 0-1 and reduce the other parts of the image.
    
    Parameters:
        image (numpy.ndarray): Input image object (OpenCV format, grayscale).
        range_min (int): Minimum value of the range to be mapped.
        range_max (int): Maximum value of the range to be mapped.
        
    Returns:
        mapped_image (numpy.ndarray): Image object after mapping the specified range.
    """
    # Create a copy of the image to modify
    mapped_image = image.copy()
    
    # Create a mask for the specified range
    mask = cv2.inRange(image, range_min, range_max)
    
    # Apply the mask to the image
    mapped_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Using a linear remap
    mapped_image = np.interp(mapped_image, (range_min, range_max), (0, 1)).astype(np.float32)

    # Set pixels below range_min to 0
    mapped_image[image < range_min] = 0
    
    # Set pixels above range_max to 65535
    mapped_image[image > range_max] = 1 
    
    return mapped_image


def noise_reduction(image, d=9, sigma_color=75, sigma_space=75):
    """
    Reduce noise in a grayscale image while preserving edges using a bilateral filter.

    Parameters:
        image (numpy.ndarray): Input grayscale image with intensity values ranging from 0 to 65,535.
        d (int): Diameter of each pixel neighborhood used during filtering. If non-positive, it's computed from sigma_space.
        sigma_color (float): Filter sigma in the color space. Larger values mean that farther colors within the pixel neighborhood will be mixed together.
        sigma_space (float): Filter sigma in the coordinate space. Larger values mean that farther pixels will influence each other.

    Returns:
        denoised_image (numpy.ndarray): Image with reduced noise.
    """
    # Check if the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image")

    # Convert image to float32 for precision and to prevent overflow
    image_float = image.astype(np.float32)

    # Apply Bilateral Filter for noise reduction
    denoised = cv2.bilateralFilter(image_float, d, sigma_color, sigma_space)

    # Clip values to the original data range and convert back to the original data type
    denoised_image = np.clip(denoised, 0, 65535).astype(image.dtype)

    return denoised_image


def reduce_poisson_noise(image, strength=10, h=0.1):
    """
    Reduce Poisson noise in a float32 image with values in range [0,1].
    Uses bilateral filtering with proper format conversion.
    
    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0,1]
        strength (int): Denoising strength (bilateral filter diameter)
        h (float): Filter strength parameter
        
    Returns:
        numpy.ndarray: Denoised image in float32 format [0,1]
    """
    # Ensure odd kernel size
    strength = strength if strength % 2 == 1 else strength + 1
    
    # First pass: work directly with float32 [0,1]
    denoised = cv2.bilateralFilter(
        image.astype(np.float32),
        strength,
        0.1,  # reduced color sigma for [0,1] range
        strength // 2  # space sigma
    )
    
    # Second pass with adjusted parameters
    denoised = cv2.bilateralFilter(
        denoised,
        strength // 2,
        0.05,  # further reduced color sigma
        strength // 4  # reduced space sigma
    )
    
    return np.clip(denoised, 0, 1).astype(np.float32)


def reduce_gaussian_noise(image, strength=10, use_gaussian=False):
    """
    Reduce Gaussian noise in a float32 image with values in range [0,1].
    Uses Non-local Means denoising with optional Gaussian blur.
    
    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0,1]
        strength (int): Denoising strength
        h (float): Filter strength parameter for NLMeans
        use_gaussian (bool): Whether to apply additional Gaussian blur
        
    Returns:
        numpy.ndarray: Denoised image in float32 format [0,1]
    """
    # Convert to uint8 for NLMeans processing (it works better in uint8)
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Apply Non-local Means denoising
    denoised = cv2.fastNlMeansDenoising(
        image_uint8,
        None,
        h=strength,
        templateWindowSize=5,
        searchWindowSize=11
    )
    
    # Optional Gaussian blur for stronger noise reduction
    if use_gaussian:
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5)
    
    # Convert back to float32 [0,1] range
    denoised = denoised.astype(np.float32) / 255.0
    
    return np.clip(denoised, 0, 1)

def wavelet_denoising(image, wavelet='db4', level=2, threshold_mode='soft'):
    """
    Apply wavelet transform denoising to a float32 image with values in range [0,1].
    
    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0,1]
        wavelet (str): Wavelet type (e.g., 'db4', 'sym4', 'coif4')
        level (int): Decomposition level
        threshold_mode (str): 'soft' or 'hard' thresholding
        
    Returns:
        numpy.ndarray: Denoised image in float32 format [0,1]
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Calculate threshold
    # Using the first diagonal detail coefficients as noise estimator
    noise_sigma = np.median(np.abs(coeffs[-1][2])) / 0.6745
    threshold = noise_sigma * np.sqrt(2 * np.log(image.size))
    
    # Apply thresholding to detail coefficients
    new_coeffs = [coeffs[0]]  # Keep the approximation coefficients unchanged
    
    # Process detail coefficients
    for i in range(1, len(coeffs)):
        # Each level has three detail coefficient arrays
        cH, cV, cD = coeffs[i]  # Horizontal, Vertical, Diagonal
        
        # Apply thresholding to each detail coefficient array
        if threshold_mode == 'soft':
            cH = pywt.threshold(cH, threshold, mode='soft')
            cV = pywt.threshold(cV, threshold, mode='soft')
            cD = pywt.threshold(cD, threshold, mode='soft')
        else:
            cH = pywt.threshold(cH, threshold, mode='hard')
            cV = pywt.threshold(cV, threshold, mode='hard')
            cD = pywt.threshold(cD, threshold, mode='hard')
            
        new_coeffs.append((cH, cV, cD))
    
    # Reconstruct the image
    denoised = pywt.waverec2(new_coeffs, wavelet)
    
    # Handle edge effects from wavelet transform
    if denoised.shape != image.shape:
        denoised = denoised[:image.shape[0], :image.shape[1]]
    
    # Ensure output is in [0,1] range
    return np.clip(denoised, 0, 1).astype(np.float32)

def bm3d_denoising(image, sigma_psd=0.1):
    """
    Apply BM3D denoising to a float32 image with values in range [0,1].
    BM3D is one of the most effective denoising algorithms that exploits
    similarity between image blocks.
    
    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0,1]
        sigma_psd (float): Noise standard deviation, higher values = stronger denoising
                         Typical values range from 0.01 to 0.2
    
    Returns:
        numpy.ndarray: Denoised image in float32 format [0,1]
    """
    if not (image.dtype == np.float32 and 0 <= image.min() <= image.max() <= 1):
        raise ValueError("Input image must be float32 with values in [0,1]")
    
    # BM3D expects values roughly in [0,1] and float32, which we already have
    denoised = bm3d.bm3d(image, sigma_psd=sigma_psd)
    
    # Ensure output is strictly in [0,1] range
    return np.clip(denoised, 0, 1).astype(np.float32)

# To get the right windows for the images, or the levels tools in Photoshop
def get_average_windows(image_files: np.array) -> tuple:
    '''
    Get the average min and max values of the images in the image_files
    The value range is between 0 and 1
    '''
    min_values = []
    max_values = []
    for image_file in image_files:

        # remove the 0 and 1 values, and use mask to get the values
        mask = (image_file > 0) & (image_file < 1)
        filtered_image = image_file[mask]

        min_values.append(filtered_image.min())
        max_values.append(filtered_image.max())

    my_min = np.percentile(min_values, 50)
    my_max = np.percentile(max_values, 50)
    return my_min, my_max


def histogram_equalization_float32(image: np.array) -> np.array:
    """
    Perform histogram equalization on a float32 image with values in the range [0, 1],
    ignoring pixels with values exactly 0 or 1.

    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0, 1].

    Returns:
        numpy.ndarray: Histogram-equalized image with values in [0, 1].
    """
    if not (image.dtype == np.float32 and image.min() >= 0.0 and image.max() <= 1.0):
        raise ValueError("Input image must be a float32 array with values in the range [0, 1].")

    # Create a mask to exclude pixels with values 0 or 1
    mask = (image > 0) & (image < 1)
    valid_pixels = image[mask]

    # Compute histogram and cumulative distribution function (CDF) for valid pixels
    hist, bin_edges = np.histogram(valid_pixels, bins=65536, range=(0, 1), density=False)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to range [0, 1]

    # Use linear interpolation to map original pixel values to equalized values
    equalized = np.zeros_like(image, dtype=np.float32)
    equalized[mask] = np.interp(valid_pixels, bin_edges[:-1], cdf_normalized)

    return equalized


def equalized_hist(img):
    return cv2.equalizeHist(img)


def clahe_float32(image: np.array, clip_limit: float = 2, tile_grid_size: tuple = (8, 8)) -> np.array:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a float32 image with values in [0, 1].

    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0, 1].
        clip_limit (float): Threshold for contrast limiting. Higher values give more contrast.
        tile_grid_size (tuple): Size of grid for histogram equalization. Tuple of (rows, cols).

    Returns:
        numpy.ndarray: CLAHE-enhanced image with values in [0, 1].
    """
    if not (image.dtype == np.float32 and image.min() >= 0.0 and image.max() <= 1.0):
        raise ValueError("Input image must be a float32 array with values in the range [0, 1].")
    
    # Convert to uint16 for CLAHE
    img_uint16 = fb.bitconverter.binary_to_grayscale_one_image(image, 'uint16')
    
    # Create CLAHE object and apply
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_uint16 = clahe.apply(img_uint16)
    
    # Convert back to float32 [0,1] range
    enhanced_float32 = fb.bitconverter.grayscale_to_binary_one_image(enhanced_uint16)
    
    return enhanced_float32

def adaptive_median_filter(image, min_size=3, max_size=7):
    """
    Apply adaptive median filter to remove salt-and-pepper noise while preserving edges.
    
    Parameters:
        image (numpy.ndarray): Input grayscale image
        min_size (int): Minimum window size (must be odd)
        max_size (int): Maximum window size (must be odd)
        
    Returns:
        numpy.ndarray: Filtered image
    """
    # Ensure window sizes are odd
    min_size = min_size if min_size % 2 == 1 else min_size + 1
    max_size = max_size if max_size % 2 == 1 else max_size + 1
    
    # Initialize output image
    filtered = np.copy(image)
    pad_size = max_size // 2
    
    # Pad the image
    padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                               cv2.BORDER_REFLECT)
    
    # Process each pixel
    for i in range(pad_size, padded.shape[0] - pad_size):
        for j in range(pad_size, padded.shape[1] - pad_size):
            window_size = min_size
            while window_size <= max_size:
                half = window_size // 2
                # Extract window
                window = padded[i-half:i+half+1, j-half:j+half+1]
                
                # Calculate statistics
                med = np.median(window)
                min_val = np.min(window)
                max_val = np.max(window)
                
                # Stage A: Check if median is impulse noise
                if min_val < med < max_val:
                    # Stage B: Check if center pixel is impulse noise
                    center = padded[i, j]
                    if min_val < center < max_val:
                        filtered[i-pad_size, j-pad_size] = center
                    else:
                        filtered[i-pad_size, j-pad_size] = med
                    break
                else:
                    # Increase window size if conditions not met
                    window_size += 2
                    if window_size > max_size:
                        filtered[i-pad_size, j-pad_size] = med
    
    return filtered

def deblock_filter(image, block_size=8, strength=1.0):
    """
    Apply deblocking filter to reduce block artifacts in compressed images.
    
    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0, 1]
        block_size (int): Size of the blocks to process (typically 8 for JPEG)
        strength (float): Strength of the deblocking filter (0.0 to 2.0)
        
    Returns:
        numpy.ndarray: Deblocked image in float32 format [0, 1]
    """
    if not (image.dtype == np.float32 and 0 <= image.min() <= image.max() <= 1):
        raise ValueError("Input image must be float32 with values in [0, 1]")
    
    height, width = image.shape
    result = image.copy()
    
    # Process horizontal boundaries
    for y in range(0, height - 1, block_size):
        for x in range(width):
            if y > 0:
                p3 = image[y-1, x]
                p2 = image[y-2, x] if y > 1 else p3
                p1 = image[y-3, x] if y > 2 else p2
                p0 = image[y-4, x] if y > 3 else p1
            else:
                p3 = p2 = p1 = p0 = image[y, x]
                
            q0 = image[y, x]
            q1 = image[y+1, x] if y < height-1 else q0
            q2 = image[y+2, x] if y < height-2 else q1
            q3 = image[y+3, x] if y < height-3 else q2
            
            # Calculate filtering strength based on local gradient
            delta = np.clip(strength * (q0 - p0), -0.5, 0.5)
            
            # Apply smoothing
            result[y-1, x] = np.clip(p3 + delta/4, 0, 1) if y > 0 else p3
            result[y, x] = np.clip(q0 - delta/4, 0, 1)
    
    # Process vertical boundaries
    for x in range(0, width - 1, block_size):
        for y in range(height):
            if x > 0:
                p3 = image[y, x-1]
                p2 = image[y, x-2] if x > 1 else p3
                p1 = image[y, x-3] if x > 2 else p2
                p0 = image[y, x-4] if x > 3 else p1
            else:
                p3 = p2 = p1 = p0 = image[y, x]
                
            q0 = image[y, x]
            q1 = image[y, x+1] if x < width-1 else q0
            q2 = image[y, x+2] if x < width-2 else q1
            q3 = image[y, x+3] if x < width-3 else q2
            
            # Calculate filtering strength based on local gradient
            delta = np.clip(strength * (q0 - p0), -0.5, 0.5)
            
            # Apply smoothing
            result[y, x-1] = np.clip(p3 + delta/4, 0, 1) if x > 0 else p3
            result[y, x] = np.clip(q0 - delta/4, 0, 1)
    
    return result

def reduce_compression_artifacts(image, strength=1.0):
    """
    Reduce compression artifacts using a combination of bilateral filtering and deblocking.
    
    Parameters:
        image (numpy.ndarray): Input float32 image with values in [0, 1]
        strength (float): Strength of artifact reduction (0.0 to 2.0)
        
    Returns:
        numpy.ndarray: Processed image with reduced artifacts in float32 format [0, 1]
    """
    # First pass: Bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(
        image.astype(np.float32),
        5,  # diameter
        0.1,  # sigma color
        3.0   # sigma space
    )
    
    # Second pass: Deblocking to reduce block artifacts
    deblocked = deblock_filter(denoised, strength=strength)
    
    # Final pass: Light bilateral filter to clean up any remaining artifacts
    final = cv2.bilateralFilter(
        deblocked,
        3,    # smaller diameter
        0.05,  # reduced sigma color
        2.0    # reduced sigma space
    )
    
    return np.clip(final, 0, 1).astype(np.float32)
