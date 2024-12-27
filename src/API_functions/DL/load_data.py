import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from typing import List

class my_Dataset(Dataset):
    def __init__(self, imagelist, labels, padding_info: pd.DataFrame=None, transform=None, preprocess=True):
        super(my_Dataset).__init__()
        self.transform = transform
        self.imagelist: List[np.ndarray] = imagelist
        self.labels: List[np.ndarray] = labels
        self.is_unlabeled = labels[0] is None if labels else False
        self.preprocess = preprocess
        self.padding_info = padding_info
        self.use_transform: bool = True

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img = self.imagelist[idx]
        label = self.labels[idx] if not self.is_unlabeled else None

        # Create padding mask if padding info is available
        mask = None
        if self.padding_info is not None:
            h, w = img.shape[:2]
            mask = np.ones((h, w), dtype=np.float32)
            pad_top = self.padding_info.iloc[idx]['padding_top']
            pad_bottom = self.padding_info.iloc[idx]['padding_bottom']
            pad_left = self.padding_info.iloc[idx]['padding_left']
            pad_right = self.padding_info.iloc[idx]['padding_right']
            
            # Set padding areas to 0 in mask
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
        
        if self.use_transform:
            if self.is_unlabeled:
                # Transform only image for unlabeled data
                augmented = self.transform(image=img)
                img = augmented['image']
                img = np.expand_dims(img, axis=0)   # Add channel dimension
                return img
            else:
                # Transform both image and mask for labeled data
                augmented = self.transform(image=img, mask=label)
                img = augmented['image']
                label = augmented['mask']
                
                transformed_mask = self.transform(image=mask)
                mask = transformed_mask['image']
                return img, label, mask
        else:
            if self.is_unlabeled:
                img = np.expand_dims(img, axis=0)
                return img
            else:
                return img, label, mask

    def set_use_transform(self, use_transform: bool):
        self.use_transform = use_transform

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

# to be continue

# def dataset_prefold(path):
#     dataset = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
#     return dataset


# def pre_fold(path):
#     train_set = dataset_prefold(os.path.join(_dataset_dir,"training"))
#     val_set = dataset_prefold(os.path.join(_dataset_dir,"validation"))
#     train_val_set = train_set + val_set

#     le = preprocessing.LabelEncoder()
#     train_val_encoded = le.fit_transform(train_val_set)

#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# You can use sklearn to split the data, and here is the self-defined function to split the data
# from sklearn.model_selection import KFold, train_test_split
# def data_devider(data: list, labels:list, ratio=0.8):

#     num_train = int(len(data) * ratio)

#     combined = list(zip(data, labels))
#     random.shuffle(combined)
#     shuffled_list1, shuffled_list2 = zip(*combined)
#     data = list(shuffled_list1)
#     labels = list(shuffled_list2)

#     train_imagelist = data[:num_train]
#     train_labels = labels[:num_train]
#     val_imagelist = data[num_train:]
#     val_labels = labels[num_train:]

#     return train_imagelist, train_labels, val_imagelist, val_labels
