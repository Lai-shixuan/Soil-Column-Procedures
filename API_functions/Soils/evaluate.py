import numpy as np

# Dice coefficient
def dice(seg: np.array, ground_true: np.array, pixel_value: int = 1) -> float:
    """
    The inputs have 2 areas, the segmented area and the background area, which must be 0.
    """ 
    dice = np.sum(seg[ground_true==pixel_value])*2.0 / (np.sum(seg) + np.sum(ground_true))
    return dice

if __name__ == '__main__':
    seg = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    ground_true = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    dice_value = dice(seg=seg, ground_true=ground_true, pixel_value=1)
    print(dice_value)
