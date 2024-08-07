import cv2
import numpy as np


def read_raw(file_path: str, width: int, height: int, depth: int, dtype=np.dtype('<i2'), virtual=True) -> np.memmap:
    """
    Read a raw file into a numpy memmap array or a list of numpy arrays!
    It may need to modify the other part of the code, dut to it is a numpy array rather than a list
    """

    if virtual:
        arr = np.memmap(file_path, dtype=dtype, mode='r', shape=(depth, width, height))
        # do not read the whole file into memory
        return [arr[i, :, :] for i in range(arr.shape[0])]

    else:
        with open(file_path, 'rb') as f:
            arr = np.fromfile(f, dtype=dtype)
            arr = arr.reshape((depth, width, height))
            return [arr[i, :, :] for i in range(arr.shape[0])]


if __name__ == '__main__':
    path = 'g:/Soil.column.0003.raw'
    width = 2325
    height = 2326
    depth = 2422

    mylist = read_raw(path, width, height, depth, dtype=np.dtype('<i2'), virtual=False)
    print(mylist[0].shape)

    cv2.imshow('image', mylist[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
