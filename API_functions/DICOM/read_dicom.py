import random
import pydicom
import matplotlib.pyplot as plt


# The point is to read the DICOM file metadata. It can be a file which has the pixel data or just the metadata of a image, or the metadata of a folder
def load_dicom(path):
    dicom_data = pydicom.dcmread(path)
    print(dicom_data)
    return dicom_data


# Read the pixel data of the DICOM file, remember to replace the tag (0x07a1, 0x100a) with the tag of the pixel data of your DICOM file
def read_data(dicom_data):
    """
    This function just to remind you the basic way to read data in dicom file
    """
    if (0x07a1, 0x100a) in dicom_data:
        image_data = dicom_data.pixel_array

        if len(image_data.shape) > 2:   # 3D image
            random_index = random.randint(0, image_data.shape[0] - 1)
            random_image = image_data[random_index]
        else:                           # 2D image
            random_image = image_data

        # 显示随机图像
        plt.imshow(random_image, cmap='gray')
        plt.title(f'Random Image Index: {random_index}')
        plt.show()
    else:
        print('No pixel data in this DICOM file')