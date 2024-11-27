import numpy as np
import cv2
import matplotlib.pyplot as plt

def padding_img(input: np.ndarray, target_size: int, color: int) -> np.ndarray:
    """
    Only for grayscale image
    """
    if input.shape[0] > target_size or input.shape[1] > target_size:
        raise ValueError("Input image size is larger than target size")

    padding_top = (target_size - input.shape[0]) // 2
    padding_left = (target_size - input.shape[1]) // 2

    output = np.pad(input, ((padding_top, target_size - input.shape[0] - padding_top),
                            (padding_left, target_size - input.shape[1] - padding_left)),
                    mode='constant', constant_values=color)
    
    return output


def restore_image(patches: list[np.ndarray], image_shape: tuple[int, int], target_size: int, stride: int) -> np.ndarray:
    """
    Reconstructs the original image from overlapping patches by averaging the overlapping areas.
    
    Parameters:
    - patches (list[np.ndarray]): List of patches.
    - image_shape (tuple[int, int]): The shape of the original image (height, width).
    - target_size (int): The size of the smaller patches (e.g., 512).
    - stride (int): The stride used during the patch extraction.
    
    Returns:
    - np.ndarray: The reconstructed image.
    """
    h, w = image_shape
    output = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    patch_idx = 0
    for y in range(0, h - target_size + 1, stride):
        for x in range(0, w - target_size + 1, stride):
            patch = patches[patch_idx]
            output[y:y + target_size, x:x + target_size] += patch
            count[y:y + target_size, x:x + target_size] += 1
            patch_idx += 1
    
    # Handle the rightmost part of the image (padding)
    if w % target_size != 0:
        for y in range(0, h - target_size + 1, stride):
            patch = patches[patch_idx]
            output[y:y + target_size, -target_size:] += patch
            count[y:y + target_size, -target_size:] += 1
            patch_idx += 1
    
    # Handle the bottom part of the image (padding)
    if h % target_size != 0:
        for x in range(0, w - target_size + 1, stride):
            patch = patches[patch_idx]
            output[-target_size:, x:x + target_size] += patch
            count[-target_size:, x:x + target_size] += 1
            patch_idx += 1
    
    # Bottom-right corner (if both dimensions aren't divisible by target_size)
    if h % target_size != 0 and w % target_size != 0:
        patch = patches[patch_idx]
        output[-target_size:, -target_size:] += patch
        count[-target_size:, -target_size:] += 1
    
    # Average out the overlapping regions
    output /= count
    return np.round(output).astype(np.uint8)


def sliding_window(input: np.ndarray, target_size: int, stride: int) -> list[np.ndarray]:
    """
    Breaks a large image into smaller patches of target_size, using a sliding window approach.
    Overlapping patches are allowed.
    
    Parameters:
    - input (np.ndarray): The input image.
    - target_size (int): The size of the smaller patches (e.g., 512).
    - stride (int): The step size for sliding the window. Typically, it's the same as the target_size, but can be smaller for overlap.
    
    Returns:
    - list[np.ndarray]: A list of smaller image patches.
    """
    patches = []
    h, w = input.shape[:2]
    
    for y in range(0, h - target_size + 1, stride):
        for x in range(0, w - target_size + 1, stride):
            patch = input[y:y + target_size, x:x + target_size]
            patches.append(patch)
    
    # Handle the rightmost part of the image (padding if necessary)
    if w % target_size != 0:
        for y in range(0, h - target_size + 1, stride):
            patch = input[y:y + target_size, -target_size:]
            patches.append(patch)
    
    # Handle the bottom part of the image (padding if necessary)
    if h % target_size != 0:
        for x in range(0, w - target_size + 1, stride):
            patch = input[-target_size:, x:x + target_size]
            patches.append(patch)
    
    # Bottom-right corner padding (if both dimensions aren't divisible by target_size)
    if h % target_size != 0 and w % target_size != 0:
        patch = input[-target_size:, -target_size:]
        patches.append(patch)
    
    return patches


# --------------------TEST--------------------

def test_padding_img():
    img = cv2.imread('g:/temp16bit/0028.384.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image file not found")

    padded_img = padding_img(img)
    plt.imshow(padded_img, cmap='gray')
    plt.show()


# --------------------MAIN--------------------

if __name__ == '__main__':
    test_padding_img()