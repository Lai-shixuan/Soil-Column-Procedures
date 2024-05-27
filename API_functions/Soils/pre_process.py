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
