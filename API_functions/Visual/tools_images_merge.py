from API_functions import file_batch as fb
import cv2
import numpy as np
import os
from natsort import natsorted
import re
import pprint

# Using paths, nature sort the file names and return the names and paths
def export_information(files):
	def clean_file_names(file_names):
		cleaned_names = []
		for file_name in file_names:
			# Remove the file extension
			base_name = os.path.splitext(file_name)[0]
			
			# Remove any text within brackets and the brackets themselves
			clean_name = re.sub(r'\(.*?\)', '', base_name)
			
			cleaned_names.append(clean_name)
	
		return cleaned_names

	temp_tuple = zip([os.path.basename(file_path) for file_path in files], files)
	temp_array = [list(elem) for elem in temp_tuple]

	file_names = natsorted(temp_array, key=lambda x: x[0])

	pprint.pp(file_names)

	names = [file_path[0] for file_path in file_names]	
	names = clean_file_names(names)
	paths = [file_path[1] for file_path in file_names]

	return names, paths


# Only for reading the suitcases images
def read_suitcases():
	path_Suitcase1 = r'e:\3.Experimental_Data\Images\Scanning-record\5.Project2-imageMerge\5.crop-Suitcase1'
	path_Suitcase1 = path_Suitcase1.replace('\\', os.sep)

	path_Suitcase2 = r'e:\3.Experimental_Data\Images\Scanning-record\5.Project2-imageMerge\6.crop-Suitcase2'
	path_Suitcase2 = path_Suitcase2.replace('\\', os.sep)

	files = fb.get_image_names(path_Suitcase1, None, 'bmp') + \
			fb.get_image_names(path_Suitcase2, None, 'bmp') + \
			fb.get_image_names(os.path.join(path_Suitcase1, 'special_png'), None, 'png')

	return export_information(files)


# Only for reading the high resolution images
def read_high_resolution():
	path_high_resolution = r'e:\3.Experimental_Data\Images\Scanning-record\5.Project2-imageMerge\4.crop-High_Resolution'
	path_high_resolution = path_high_resolution.replace('\\', os.sep)
	
	path_Suitcase1 = r'e:\3.Experimental_Data\Images\Scanning-record\5.Project2-imageMerge\5.crop-Suitcase1'
	path_Suitcase1 = path_Suitcase1.replace('\\', os.sep)

	path_Suitcase2 = r'e:\3.Experimental_Data\Images\Scanning-record\5.Project2-imageMerge\6.crop-Suitcase2'
	path_Suitcase2 = path_Suitcase2.replace('\\', os.sep)

	files = fb.get_image_names(path_high_resolution, None, 'bmp') + \
			fb.get_image_names(path_Suitcase1, None, 'bmp') + \
			fb.get_image_names(path_Suitcase2, None, 'bmp') + \
			fb.get_image_names(os.path.join(path_Suitcase1, 'special_png'), None, 'png')

	files = [file for file in files if os.path.basename(file).startswith(('5', '15', '25'))]

	return export_information(files)
	

def merge_images(image_paths, rows, columns):
	# Load all images, and turn to grayscale
	images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
	images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
	
	if rows * columns != len(images):
		raise ValueError('The number of images must be equal to rows * lengths')

	single_height, single_width = images[0].shape[:2]
	total_width = single_width * columns
	total_height = single_height * rows
	
	# Create a new image (numpy array) with the appropriate size
	merged_image = np.zeros((total_height, total_width), dtype=np.uint8)
	
	# Copy images into the new image
	for index, image in enumerate(images):
		x_offset = (index % columns) * single_width
		y_offset = (index // columns) * single_height
		merged_image[y_offset:y_offset+single_height, x_offset:x_offset+single_width] = image
	
	return merged_image


def write_texts(text_array, rows, columns, image):
	def get_text_center_position(text, font, font_scale, thickness, position, box_width, box_height):
		text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
		text_x = position[0] + (box_width - text_size[0]) // 2
		text_y = position[1] + box_height - 2 * text_size[1]
		return (text_x, text_y)
	
	if rows * columns != len(text_array):
		raise ValueError('The number of images must be equal to rows * lengths')

	box_width = 170
	box_height = 1242
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.5 
	color = (255, 255, 255)  # White color
	thickness = 1

	for i, text in enumerate(text_array):
		col = i % columns
		row = i // columns
		position = (col * box_width, row * box_height)
		text_position = get_text_center_position(text, font, font_scale, thickness, position, box_width, box_height)
		cv2.putText(image, text, text_position, font, font_scale, color, thickness)

	return image


if __name__ == '__main__':
	names_30um, paths_30um = read_high_resolution()
	merged_image = merge_images(paths_30um, rows=3, columns=4)
	write_texts_image = write_texts(names_30um, rows=3, columns=4, image=merged_image)
	cv2.imwrite('./output_image1.jpg', merged_image)

	names_suitcases, paths_suitcases = read_suitcases()
	merged_image = merge_images(paths_suitcases, rows=2, columns=34)
	write_texts_image = write_texts(names_suitcases, rows=2, columns=34, image=merged_image)
	cv2.imwrite('./output_image2.jpg', merged_image)

