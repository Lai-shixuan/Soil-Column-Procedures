import numpy as np
import cv2
import matplotlib.pyplot as plt

def padding_img(input: np.ndarray, target_size: int=512) -> np.ndarray:
    """
    Only for grayscale image
    """
    if input.shape[0] > target_size or input.shape[1] > target_size:
        raise ValueError("Input image size is larger than target size")

    padding_top = (target_size - input.shape[0]) // 2
    padding_left = (target_size - input.shape[1]) // 2

    output = np.pad(input, ((padding_top, target_size - input.shape[0] - padding_top),
                            (padding_left, target_size - input.shape[1] - padding_left)),
                    mode='constant', constant_values=255)
    
    return output

if __name__ == '__main__':
    img = cv2.imread('g:/temp16bit/0028.384.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image file not found")

    padded_img = padding_img(img)
    plt.imshow(padded_img, cmap='gray')
    plt.show()