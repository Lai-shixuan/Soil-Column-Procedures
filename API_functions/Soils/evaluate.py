import numpy as np
import torch
from typing import Union


def dice_coefficient(seg: Union[np.ndarray, torch.Tensor], ground_true: Union[np.ndarray, torch.Tensor], pixel_value: int = 1) -> float:

    """
    Calculate the Dice coefficient. Automatically detects if the inputs are NumPy arrays or PyTorch tensors.
    The inputs have 2 areas, the segmented area and the background area, which must be 0.
    Only works for 2 classes (background and segmented area).
    """
    if isinstance(seg, np.ndarray) and isinstance(ground_true, np.ndarray):
        # NumPy calculation
        dice = np.sum(seg[ground_true == pixel_value]) * 2.0 / (np.sum(seg) + np.sum(ground_true))
    elif isinstance(seg, torch.Tensor) and isinstance(ground_true, torch.Tensor):
        # PyTorch calculation
        dice = 2.0 * torch.sum(seg[ground_true == pixel_value]) / (seg.sum() + ground_true.sum())
        dice = dice.item()
    else:
        raise ValueError("Input types must be both NumPy arrays or both PyTorch tensors.")
    
    return dice


def iou(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor], n_classes: int = 2) -> Union[float, np.ndarray]:
    """
    Calculate the Intersection over Union (IoU). Automatically detects if the inputs are NumPy arrays or PyTorch tensors.
    The inputs have 2 areas, the segmented area and the background area, which must be 0.
    Support multiple classes.
    """
    ious = []

    # Determine the type of the inputs
    if isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        flatten = lambda x: x.ravel()
        sum_func = np.sum
        logical_and = np.logical_and
    elif isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        flatten = lambda x: x.view(-1)
        sum_func = torch.sum
        logical_and = lambda a, b: a & b
    else:
        raise ValueError("Input types must be both NumPy arrays or both PyTorch tensors.")

    pred = flatten(pred)
    target = flatten(target)

    # Ignore IoU for background class ("0")
    for class_num in range(1, n_classes):
        pred_indexes = pred == class_num
        target_indexes = target == class_num

        intersection = sum_func(logical_and(pred_indexes, target_indexes))
        union = sum_func(pred_indexes) + sum_func(target_indexes) - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    if len(ious) == 1:
        return ious[0]
    else:
        return np.array(ious) if isinstance(pred, np.ndarray) else torch.tensor(ious, dtype=torch.float32)
    

def mIoU(pred, target, n_classes = 2):
    ious = iou(pred, target, n_classes)
    return np.nanmean(ious)


if __name__ == '__main__':
    seg = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    ground_true = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    dice_value = dice_coefficient(seg=seg, ground_true=ground_true, pixel_value=1)
    print(f'dice_value={dice_value}')
    
    iou_value = iou(pred=seg, target=ground_true, n_classes=2)
    print(f'iou_value={iou_value}')

    miou_value = mIoU(pred=seg, target=ground_true, n_classes=2)
    print(f'miou_value={miou_value}')

    seg = torch.tensor(seg)
    ground_true = torch.tensor(ground_true)
    tensor = dice_coefficient(seg=seg, ground_true=ground_true, pixel_value=1)
    print(f'tensor={tensor}')

    iou_value = iou(pred=seg, target=ground_true, n_classes=2)
    print(f'iou_value={iou_value}')

    miou_value = mIoU(pred=seg, target=ground_true, n_classes=2)
    print(f'miou_value={miou_value}')
