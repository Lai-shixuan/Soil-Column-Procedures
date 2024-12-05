import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
# sys.path.insert(0, "/root/Soil-Column-Procedures")
from src.API_functions.Images import file_batch as fb


def origin(numbers: np.ndarray):
    return numbers


# non-parameter methods
def otsu(numbers: np.ndarray):
    if numbers.max() > 255:
        _, dst = cv2.threshold(numbers, 0, 65535, cv2.THRESH_OTSU)
    elif numbers.max() > 1:
        _, dst = cv2.threshold(numbers, 0, 255, cv2.THRESH_OTSU)
    elif numbers.max() <= 1:
        _, dst = cv2.threshold(numbers, 0, 1, cv2.THRESH_OTSU)
    return dst


def kapur_entropy_3d(image: np.ndarray):
    shape = image.shape

    # Calculate histogram of the image
    image = image.reshape(-1, 1)
    image = np.uint8(image)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()

    # Compute cumulative sum and cumulative mean
    cum_sum = hist.cumsum()

    # Initialize variables
    max_ent = 0.0
    optimal_threshold = 0.0
    small_value = 1e-5

    t_list = []

    entropy_list = []

    # kapur's entropy method
    for t in range(256):
        omega_a = cum_sum[t]
        if omega_a > 0.01 and 1 - omega_a > 0.01:
            t_list.append(t)

    hist = np.where(hist == 0, small_value, hist)

    for t in t_list:
        omega_a = cum_sum[t]
        omega_b = cum_sum[-1] - omega_a
        # Calculate entropy
        entropy_a = -np.sum(hist[t_list[0]:t] / omega_a * np.log(hist[t_list[0]:t] / omega_a))
        entropy_b = -np.sum(hist[t + 1:t_list[-1]] / omega_b * np.log(hist[t + 1:t_list[-1]] / omega_b))
        total_entropy = entropy_a + entropy_b
        entropy_list.append([t, total_entropy])

        # Check if new maximum entropy is found
        if total_entropy > max_ent:
            max_ent = total_entropy
            optimal_threshold = t

    # Apply threshold to the image
    _, threshold_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)

    return threshold_image


def kmeans_3d(numbers: np.ndarray, mask: np.ndarray = None):
    """
    K-means clustering for 3D image with consistent labeling.
    The brighter pixels will always be labeled as 1, darker as 0.
    """
    original_shape = numbers.shape
    
    if numbers.max() > 255:
        type = '16bit'
        numbers = numbers / 65535
    elif numbers.max() > 1:
        type = '8bit'
        numbers = numbers / 255
    else:
        type = 'float'
        numbers = numbers

    if mask is not None:
        # Only cluster masked pixels
        masked_values = numbers[mask].reshape(-1, 1)
        kmeans_filter = KMeans(n_clusters=2, random_state=0, n_init=10).fit(masked_values)
        
        # Check if labels need to be flipped based on cluster centers
        centers = kmeans_filter.cluster_centers_.flatten()
        labels = kmeans_filter.labels_
        if centers[1] > centers[0]:
            # Flip labels if cluster 0 is brighter than cluster 1
            labels = 1 - labels
        
        # Create output array filled with zeros
        classes = np.zeros_like(numbers)
        # Assign clustered values back to masked positions
        classes[mask] = labels
    else:
        numbers_reshaped = numbers.reshape(-1, 1)
        kmeans_filter = KMeans(n_clusters=2, random_state=0, n_init=10).fit(numbers_reshaped)
        
        # Check if labels need to be flipped based on cluster centers
        centers = kmeans_filter.cluster_centers_.flatten()
        labels = kmeans_filter.labels_
        if centers[1] > centers[0]:
            # Flip labels if cluster 0 is brighter than cluster 1
            labels = 1 - labels
            
        classes = labels.reshape(original_shape)

    if type == '16bit':
        classes = classes * 65535
    elif type == '8bit':
        classes = classes * 255
    else:
        classes = classes

    return classes


def gmm_3d(numbers: np.ndarray):
    shape = numbers.shape
    numbers = numbers.reshape(-1, 1)
    numbers = numbers / 255
    gmm_filter = GaussianMixture(n_components=2, random_state=0).fit(numbers)
    pixels_gmm = gmm_filter.predict(numbers)
    pixels_gmm = pixels_gmm.reshape(shape) * 255
    return pixels_gmm


def user_threshold(image: np.ndarray, optimal_threshold: float):
    """Apply threshold with value in 0-1 range.
    
    Args:
        image: Input image array
        optimal_threshold: Threshold value in range [0,1]
        
    Returns:
        Binary image
    """
    # Convert threshold to image range
    if image.max() > 255:
        threshold = optimal_threshold * 65535
        _, threshold_image = cv2.threshold(image, threshold, 65535, cv2.THRESH_BINARY)
    elif image.max() > 1:
        threshold = optimal_threshold * 255
        _, threshold_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif image.max() <= 1:
        _, threshold_image = cv2.threshold(image, optimal_threshold, 1, cv2.THRESH_BINARY)
    else:
        raise ValueError('Wrong value range of image')
    return threshold_image


# ----------------- Test -----------------


def test_user_threshold():
    imgs_path = fb.get_image_names('g:/DL_Data_raw/Unit_test/threshold/temp16bit/', None, 'png')
    imgs_names = [os.path.basename(p) for p in imgs_path]
    imgs = fb.read_images(imgs_path, gray='gray', read_all=True)
    for i, img in enumerate(imgs):
        thresholded_img = user_threshold(img, 45621)
        thresholded_img = 65535 - thresholded_img
        cv2.imwrite(f'g:/DL_Data_raw/Unit_test/threshold/threshold/{imgs_names[i]}', thresholded_img)


# ----------------- Main -----------------


if __name__ == '__main__':
    test_user_threshold()