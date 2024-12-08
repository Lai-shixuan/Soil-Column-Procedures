import cv2
import numpy as np

def read_raw(file_path: str, height: int, width: int, depth: int, dtype=np.dtype('<i2'), virtual=True) -> np.ndarray:
    """
    Read a raw file into a numpy memmap array or a numpy array.
    """

    if virtual:
        arr = np.memmap(file_path, dtype=dtype, mode='r', shape=(depth, height, width))
        # do not read the whole file into memory
        return arr

    else:
        with open(file_path, 'rb') as f:
            arr = np.fromfile(f, dtype=dtype)
            arr = arr.reshape((depth, height, width))
            return arr

if __name__ == '__main__':
    path = 'g:/Soil.column.0003.raw'
    width = 2325
    height = 2326
    depth = 2422

    myarray = read_raw(path, width, height, depth, dtype=np.dtype('<i2'), virtual=False)
    print(myarray.shape)

    cv2.imshow('image', myarray[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
