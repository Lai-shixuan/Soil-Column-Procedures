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


def equalized_hist(img):
    return cv2.equalizeHist(img)


def median(img, kernel_size: int=3):
    return cv2.medianBlur(img, kernel_size)


# Only for 2D image, that is, grayscale image
def image_cut(img, x0, y0, edge_length):
    return img[x0 - int(1 / 2 * edge_length): x0 + int(1 / 2 * edge_length),
           y0 - int(1 / 2 * edge_length): y0 + int(1 / 2 * edge_length)]


# RGB to grayscale, whether it is a RGB image or not
def rgb2gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        return img


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
        image (numpy.ndarray): Input grayscale image object.
        sigma (float): Standard deviation for Gaussian blur, which controls the smoothness level.
        k (float): Boost factor for enhancing high-frequency components. Higher values increase the edge enhancement effect.

    Returns:
        enhanced_image (numpy.ndarray): Image with enhanced edges and details.
    """
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Calculate the high-pass component (original - blurred)
    high_pass = cv2.subtract(image, blurred)
    
    # Enhance the image by adding the high-pass component, scaled by factor k
    enhanced_image = cv2.addWeighted(image, 1 + k, high_pass, k, 0)
    
    return enhanced_image


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