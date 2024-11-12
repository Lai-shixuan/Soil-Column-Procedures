import sys
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain")

from API_functions import file_batch as fb
import os
import cv2
from tqdm import tqdm
import os
import numpy as np

def convert_to_8bit(path_in, path_out):

    # Create the output directory if it does not exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Get all image files in the input directory
    image_names = fb.get_image_names(path_in, None, 'png')
    names = [os.path.basename(item) for item in image_names]
    image_files = fb.read_images(image_names, gray='gray', read_all=True)

    # min = 255
    # max = 0
    # for image_file in image_files:
    #     if image_file.min() < min:
    #         min = image_file.min()
    #     if image_file.max() > max:
    #         max = image_file.max()

    for i, image_file in tqdm(enumerate(image_files)):
        
        min = image_file.min()
        max = image_file.max()
        image_8bit = ((image_file - min) / (max - min) * 255).astype(np.uint8)

        # # Convert the image to 8-bit
        # image_8bit = (image_file / 256).astype('uint8')

        # # Normalize the image to 0-255
        # image_8bit = cv2.normalize(image_8bit, None, 0, 255, cv2.NORM_MINMAX)

        # Save the 8-bit image to the output directory
        output_path = os.path.join(path_out, names[i])
        cv2.imwrite(output_path, image_8bit)

    print("\033[1;3mConversion to 8-bit completed!\033[0m")

if __name__ == '__main__':
    path_in = 'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.0028/16bits/1.Reconstruct/'
    path_out = 'f:/3.Experimental_Data/Soils/Quzhou_Henan/Soil.column.0028/8bits/1.Reconstruct/'
    convert_to_8bit(path_in, path_out)