import os
import cv2
import matplotlib.pyplot as plt


# A function to read an image using opencv
def read_image_opencv(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise Exception('Error: Image not found')
    print("\033[1;3mReading Completed!\033[0m")
    return img


# A function to convert opencv image to PIL image
def convert_cv2_to_pil(cv2_image):
    pil_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return pil_image


# A function to show a image using matplotlib, whether it is a grayscale or a color image
def show_image(img):
    if img.shape.__len__() == 3:
        img = convert_cv2_to_pil(img)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif img.shape.__len__() == 2:
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()


# A function to save the image in specific format, using plt:
def save_image(image, image_path: str, name: str, image_format: str):
    cv2.imwrite(os.path.join(image_path, name + '.' + image_format), image)
    print("\033[1;3mSaving Completed!\033[0m")
