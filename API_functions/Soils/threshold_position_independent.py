import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def origin(numbers: np.ndarray):
    return numbers


# non-parameter methods
def otsu(numbers: np.ndarray):
    if numbers.max() > 255:
        _, dst = cv2.threshold(numbers, 0, 65535, cv2.THRESH_OTSU)
    else:
        _, dst = cv2.threshold(numbers, 0, 255, cv2.THRESH_OTSU)
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


def kmeans_3d(numbers: np.ndarray):
    """
    K-means clustering for 3D image, and 16bit image is supported.
    """
    if numbers.max()  > 65535:
        bit16 = True
        numbers = numbers / 65535
    else:
        bit16 = False
        numbers = numbers / 255

    shape = numbers.shape
    numbers = numbers.reshape(-1, 1)
    kmeans_filter = KMeans(n_clusters=2, random_state=0, n_init=10).fit(numbers)
    classes = kmeans_filter.labels_
    classes = classes.reshape(shape)

    if bit16:
        classes = classes * 65535
    else:
        classes = classes * 255
    return classes


def gmm_3d(numbers: np.ndarray):
    shape = numbers.shape
    numbers = numbers.reshape(-1, 1)
    numbers = numbers / 255
    gmm_filter = GaussianMixture(n_components=2, random_state=0).fit(numbers)
    pixels_gmm = gmm_filter.predict(numbers)
    pixels_gmm = pixels_gmm.reshape(shape) * 255
    return pixels_gmm


def user_threshold(image: np.ndarray, optimal_threshold: int):
    if image.max() > 255:
        _, threshold_image = cv2.threshold(image, optimal_threshold, 65535, cv2.THRESH_BINARY)
    else:
        _, threshold_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)
    return threshold_image


# ----------------- Test -----------------


def test_user_threshold():
    img = cv2.imread('/root/Soil-Column-Procedures/data/version1/test_images/0028.387.png', cv2.IMREAD_GRAYSCALE)
    thresholded_img = kmeans_3d(img)
    cv2.imwrite('/root/Soil-Column-Procedures/data/version1/test_labels/0028.387.png', thresholded_img)


# ----------------- Main -----------------


if __name__ == '__main__':
    test_user_threshold()