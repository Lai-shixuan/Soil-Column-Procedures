import numpy as np
import torch
from typing import Union
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-4):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)
        
        # BCE Loss with mask (using BCEWithLogitsLoss)
        bce = self.bce(pred, target)
        bce = (bce * mask).sum() / mask.sum()
        
        # Dice Loss using soft_dice_coefficient
        # Need sigmoid for dice calculation since we're using logits for BCE
        pred_sigmoid = torch.sigmoid(pred)
        dice = 1 - soft_dice_coefficient(target, pred_sigmoid, mask, self.smooth)
        
        return bce * 0.2 + dice * 0.8


def soft_dice_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor = None, smooth=1e-6) -> torch.Tensor:
    """
    Calculate the soft Dice coefficient. The inputs are PyTorch tensors.
    Args:
        y_true: Ground truth tensor
        y_pred: Prediction tensor
        mask: Optional mask tensor (1 for valid regions, 0 for padding)
        smooth: Smoothing factor to avoid division by zero
    """
    if mask is None:
        mask = torch.ones_like(y_true)
        
    y_true = (y_true * mask).view(-1)
    y_pred = (y_pred * mask).view(-1)
    mask = mask.view(-1)
    
    intersection = (y_true * y_pred * mask).sum()
    total = ((y_true ** 2) * mask).sum() + ((y_pred ** 2) * mask).sum()
    dice = (2. * intersection + smooth) / (total + smooth)
    
    return dice


def dice_coefficient(seg: Union[np.ndarray, torch.Tensor], ground_true: Union[np.ndarray, torch.Tensor], pixel_value: int = 1) -> float:

    """
    Calculate the Dice coefficient. Automatically detects if the inputs are NumPy arrays or PyTorch tensors.
    The inputs have 2 areas, the segmented area and the background area, which must be 0.
    Only works for 2 classes (background and segmented area).
    """

    if isinstance(seg, np.ndarray) and isinstance(ground_true, np.ndarray):
        # NumPy calculation
        seg = (seg >= 0.5).astype(np.int16)
        dice = np.sum(seg[ground_true == pixel_value]) * 2.0 / (np.sum(seg) + np.sum(ground_true))
    elif isinstance(seg, torch.Tensor) and isinstance(ground_true, torch.Tensor):
        # PyTorch calculation
        seg = (seg >= 0.5).int()
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
    The return value is a float if there is only one class, otherwise it is a NumPy array.
    """
    ious = []

    # Determine the type of the inputs
    if isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        pred = (pred >= 0.5).astype(np.int16)
        flatten = lambda x: x.ravel()
        sum_func = np.sum
        logical_and = np.logical_and
    elif isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = (pred >= 0.5).int()
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


# Calculate the mean IoU if there are multiple classes
def mIoU(pred, target, n_classes = 2):
    ious = iou(pred, target, n_classes)
    return np.nanmean(ious)


def f1_score(pred, gt):
    # Check the input type
    if isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray):
        pred_binary = (pred >= 0.5)  # Binarize predictions
        gt_binary = (gt == 1)       # Ensure ground truth is binary
        
        # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = np.logical_and(pred_binary, gt_binary).sum()
        FP = np.logical_and(pred_binary, np.logical_not(gt_binary)).sum()
        FN = np.logical_and(np.logical_not(pred_binary), gt_binary).sum()
    elif torch.is_tensor(pred) and torch.is_tensor(gt):
        pred_binary = (pred >= 0.5)  # Binarize predictions
        gt_binary = (gt == 1)       # Ensure ground truth is binary
        
        TP = torch.logical_and(pred_binary, gt_binary).sum().item()
        FP = torch.logical_and(pred_binary, torch.logical_not(gt_binary)).sum().item()
        FN = torch.logical_and(torch.logical_not(pred_binary), gt_binary).sum().item()
    else:
        # Raise an error if inputs are not both NumPy arrays or PyTorch tensors
        raise TypeError("Inputs must both be NumPy arrays or PyTorch tensors")
    
    # Calculate Precision and Recall
    TP_FP = TP + FP  # Total predicted positives
    TP_FN = TP + FN  # Total actual positives
    if TP_FP == 0:
        precision = 0.0  # Handle case where precision denominator is zero
    else:
        precision = TP / TP_FP
    if TP_FN == 0:
        recall = 0.0  # Handle case where recall denominator is zero
    else:
        recall = TP / TP_FN
    
    # Calculate F1 Score
    if precision + recall == 0:
        f1 = 0.0  # Handle case where both precision and recall are zero
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1


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
