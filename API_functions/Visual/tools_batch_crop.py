from API_functions import file_batch as fb
import cv2
import os

path = r'e:\3.Experimental_Data\Images\Scanning-record\5.Project2-imageMerge\1.High_Resolution'
path = path.replace('\\', os.sep)
files = fb.get_image_names(path, None, 'bmp')

for file in files:
    # read the image with bmp format
    img = cv2.imread(file)
    img = img[4:, 910:1080]

    # change the save path to a sub folder named 'crop'
    file = file.replace('1.High_Resolution', '4.crop-High_Resolution')
    cv2.imwrite(file, img)
