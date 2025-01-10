import torch
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
from torch.utils.data import Dataset, Sampler

import sys
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")
from src.workflow_tools.database import s4augmented_labels

class my_Dataset(Dataset):
    def __init__(self, imagelist: List[np.ndarray], labels: List[np.ndarray], padding_info: pd.DataFrame=None, transform=None):
        super(my_Dataset).__init__()
        self.imagelist = imagelist
        self.labels = labels
        self.is_unlabeled = torch.tensor([label is None for label in labels])
        self.padding_info = padding_info

        self.use_transform: bool = True
        self.teacher_model = None

        self.transform = transform
        if self.transform is None:
            self.use_transform = False

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        img = self.imagelist[idx]
        if self.is_unlabeled[idx]:
            img_tensor = torch.from_numpy(img).float()
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to('cuda')
            label = self.teacher_model(img_tensor)
            label = torch.sigmoid(label).to('cpu').detach().squeeze(0).squeeze(0).numpy()
        else:
            label = self.labels[idx]

        if self.padding_info is not None:
            mask = self.create_mask(img, idx)
        
        if self.use_transform:
            augmenter = s4augmented_labels.ImageAugmenter(img, label, mask=mask)
            augmented_img, augmented_label, _ = augmenter.augment()

            augmented = self.transform(image=augmented_img, masks=[augmented_label, mask])
            return augmented['image'], augmented['masks'][0], augmented['masks'][1], self.is_unlabeled[idx]
        else:
            print("Warning, no transform is applied to the dataset. And they are numpy arrays.")
            return img, label, mask, self.is_unlabeled[idx]

    def create_mask(self, img, idx) -> np.ndarray:
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        pad_top = self.padding_info.iloc[idx]['padding_top']
        pad_bottom = self.padding_info.iloc[idx]['padding_bottom']
        pad_left = self.padding_info.iloc[idx]['padding_left']
        pad_right = self.padding_info.iloc[idx]['padding_right']
        
        if pad_top > 0:
            pad_top = int(pad_top)
            mask[:pad_top, :] = 0
        if pad_bottom > 0:
            pad_bottom = int(pad_bottom)
            mask[-pad_bottom:, :] = 0
        if pad_left > 0:
            pad_left = int(pad_left)
            mask[:, :pad_left] = 0
        if pad_right > 0:
            pad_right = int(pad_right)
            mask[:, -pad_right:] = 0
        
        return mask
    
    def set_use_transform(self, use_transform: bool):
        self.use_transform = use_transform

    def set_teacher_model(self, teacher_model):
        self.teacher_model = teacher_model

    def update_label_by_index(self, idx, pred, threshold=0.8):
        original_label = torch.from_numpy(self.labels[idx]).to('cuda')

        # Create update masks for both foreground and background
        update_mask_fg = pred > threshold
        update_mask_bg = pred < (1 - threshold)

        new_label = original_label.clone()
        new_label[update_mask_fg] = 1  # Update confident foreground predictions
        new_label[update_mask_bg] = 0  # Update confident background predictions

        # Safety check to prevent over-updates
        h, w = original_label.shape
        total_pixels = h * w
        new_foreground_ratio = torch.sum(new_label != 0).item() / total_pixels
        original_foreground_ratio = torch.sum(original_label != 0).item() / total_pixels

        # Only update if the change is not too dramatic
        if abs(new_foreground_ratio - original_foreground_ratio) < 0.3:
            self.labels[idx] = new_label.cpu().numpy()


class MixedRatioSampler(Sampler):
    def __init__(self, data_set: my_Dataset, labeled_ratio: float, batch_size: int):
        """
        Initialize the mixed ratio sampler
        Args:
            data_set: Dataset containing both labeled and unlabeled data
            labeled_ratio: Ratio of labeled data in each batch (0-1)
            batch_size: Size of each batch
        """
        self.data_source = data_set
        self.batch_size = batch_size
        if not (0 <= labeled_ratio <= 1):
            raise ValueError("labeled_ratio must be between 0 and 1")
        
        self.labeled_indices = torch.where(~data_set.is_unlabeled)[0]
        self.unlabeled_indices = torch.where(data_set.is_unlabeled)[0]
        
        self.labeled_ratio = labeled_ratio
        self.labeled_per_batch = int(self.batch_size * self.labeled_ratio)
        self.unlabeled_per_batch = self.batch_size - self.labeled_per_batch
        
        if len(self.labeled_indices) == 0 or len(self.unlabeled_indices) == 0:
            raise ValueError("Both labeled and unlabeled data must exist in the dataset")

        # Calculate number of complete batches
        self.num_batches = min(
            len(self.labeled_indices) // self.labeled_per_batch,
            len(self.unlabeled_indices) // self.unlabeled_per_batch
        )

    def __iter__(self):
        # Shuffle indices
        labeled_indices = self.labeled_indices[torch.randperm(len(self.labeled_indices))]
        unlabeled_indices = self.unlabeled_indices[torch.randperm(len(self.unlabeled_indices))]
        
        # Generate batches
        all_indices = []
        for i in range(self.num_batches):
            # Get indices for current batch
            batch_labeled = labeled_indices[i * self.labeled_per_batch : (i + 1) * self.labeled_per_batch]
            batch_unlabeled = unlabeled_indices[i * self.unlabeled_per_batch : (i + 1) * self.unlabeled_per_batch]
            
            # Combine and shuffle the batch
            batch = torch.cat([batch_labeled, batch_unlabeled])
            batch = batch[torch.randperm(len(batch))]
            all_indices.append(batch)
        
        # Flatten and return iterator
        all_indices = torch.cat(all_indices)
        return iter(all_indices.tolist())

    def __len__(self):
        return self.num_batches * self.batch_size