import cv2
import numpy as np


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


if __name__ == '__main__':
    path = 'f:/3.Experimental_Data/Soils/temp/100_origin_segmented0009.png'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    enhanced_image = high_boost_filter(image, sigma=1, k=1.2)
    # enhanced_image = laplacian_edge_enhancement(image, alpha=1)
    
    # Example usage of map_range_to_255
    range_min = 54
    range_max = 126
    mapped_image = map_range_to_255(image, range_min, range_max)
    mapped_enhanced_image = map_range_to_255(enhanced_image, range_min, range_max)
    
    # Create scalable windows
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Enhanced Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mapped Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mapped Enhanced Image', cv2.WINDOW_NORMAL)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', enhanced_image)
    cv2.imshow('Mapped Image', mapped_image)
    cv2.imshow('Mapped Enhanced Image', mapped_enhanced_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()