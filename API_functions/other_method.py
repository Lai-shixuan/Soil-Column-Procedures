import cv2
import os
import glob
import file_batch as fb


# A function to convert to specific format
def format_conversion(folder_path, input_image_name, output_image_name, input_format, output_format):
    image_names = fb.get_image_names(folder_path, input_image_name, format)





# Function to convert BMP images to PNG format
# def convert_to_(folder_path):
#     search_path = os.path.join(folder_path, '*.bmp')
#     bmp_files = glob.glob(search_path)
#
#     for idx, bmp_file in enumerate(bmp_files):
#         image = cv2.imread(bmp_file, cv2.IMREAD_GRAYSCALE)
#         new_file_name = f"1_png_{idx:03d}.png"
#         new_file_path = os.path.join(folder_path, new_file_name)
#         cv2.imwrite(new_file_path, image)
#
#
# # Function to convert images to binary format
# def convert_to_binary(folder_path):
#     search_path = os.path.join(folder_path, '*.png')
#     png_files = glob.glob(search_path)
#
#     for idx, png_file in enumerate(png_files):
#         image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
#         if image is not None:
#             binary_image = image // 255
#             new_file_name = f"2_binary_{idx:03d}.png"
#             new_file_path = os.path.join(folder_path, new_file_name)
#             cv2.imwrite(new_file_path, binary_image)
#         else:
#             print(f"Image not found: {png_file}")


# Function to convert binary images to grayscale
def binary_to_grayscale(folder_path):
    search_path = os.path.join(folder_path, '*.png')
    binary_files = glob.glob(search_path)

    for idx, binary_file in enumerate(binary_files):
        image = cv2.imread(binary_file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            grayscale_image = image * 255
            new_file_name = f"3_grayscale_{idx:03d}.png"
            new_file_path = os.path.join(folder_path, new_file_name)
            cv2.imwrite(new_file_path, grayscale_image)
        else:
            print(f"Image not found: {binary_file}")
