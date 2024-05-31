import numpy as np
import cv2
import matplotlib.pyplot as plt

class zoom_region:
    """
    A class to represent a zoom region in an image. It have 2 kinds of points: (y, x) for numpy and (x, y) for cv2.
    """
    
    def __init__(self, y:int, x:int, width:int, height:int):
        self.y = y
        self.x = x
        self.height = height
        self.width = width
        self.point1 = (y, x)
        self.point2 = (y+height, x+width)
        self.cv_point1 = (x, y)
        self.cv_point2 = (x+width, y+height)


def zoom_in(image:np.ndarray, zoom: zoom_region):
    return image[zoom.y:zoom.y+zoom.height, zoom.x:zoom.x+zoom.width]


def images_differnce(image1:np.ndarray, image2:np.ndarray, x:int, y:int, size:int):
    zoom1 = zoom_in(image1, x, y, size)
    zoom2 = zoom_in(image2, x, y, size)
    return np.sum(np.abs(zoom1-zoom2))

