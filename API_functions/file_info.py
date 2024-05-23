import cv2
import matplotlib.pyplot as plt
import numpy as np


# get the size, color style of the image
def get_info(img):
    if img is None:
        raise Exception('Error: Image not found')
    print(f"Image size: {img.shape}")
    if img.shape.__len__() == 2:
        if is_binary_image(img):
            print("Image is binary")
        else:
            print("Image is grayscale, not binary image")
    elif img.shape.__len__() == 3:
        print("Image is BGR color")
    else:
        raise Exception('Error: Image is not grayscale or BGR color')
    print("\033[1;3mGetting information completed!\033[0m")


def calculate_hist(image):
    if image.shape.__len__() == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist
    elif image.shape.__len__() == 3:
        color = ('b', 'g', 'r')
        hist = []
        for i, col in enumerate(color):
            hist.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        return hist


def plot_hist(hist, total=None, index : tuple=None):
    """
    Display a histogram using matplotlib.
    Args:
    - hist (numpy array): The histogram to display.
    - total (int, optional): Total number of histograms to be displayed in subplots.
    - index (tuple, optional): Index of the subplot where the histogram will be displayed.
    """

    def plot(hist):
        if hist.__len__() == 2:
            plt.plot(hist, color='black')
            plt.xlim([0, 256])
        elif hist.__len__() == 3:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                plt.plot(hist[i], color=col)
                plt.xlim([0, 256])

    if total is not None and index is not None:
        plt.subplot(index[0], index[1], index[2])
        plot(hist)

    if total is None or (total is not None and index == total):
        plot(hist)
    
    
def is_binary_image(img):
    unique_values = np.unique(img)
    return (np.array_equal(unique_values, [0, 255])
            or np.array_equal(unique_values, [0])
            or np.array_equal(unique_values, [255]))


def calculate_pore_percentage(image):
    if len(image.shape) != 2:
        raise Exception('Error: The image is not a grayscale image')
    elif is_binary_image(image) is False:
        raise Exception('Error: The image is not a binary image')
    else:
        pore = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == 0:
                    pore += 1
        return pore / (image.shape[0] * image.shape[1]) * 100
