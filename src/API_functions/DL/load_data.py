from torch.utils.data import Dataset

class my_Dataset(Dataset):
    def __init__(self, imagelist, labels, transform=None, preprocess=True):
        super(my_Dataset).__init__()
        self.transform = transform
        self.imagelist = imagelist
        self.labels = labels
        self.is_unlabeled = labels[0] is None if labels else False
        self.preprocess = preprocess

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img = self.imagelist[idx]
        label = self.labels[idx] if not self.is_unlabeled else None

        if self.preprocess:
            pass
            # Apply median filter and histogram equalization
            # Apply mean normalization
            # img = pre_process.median(img, kernel_size=3)
            # img = pre_process.clahe_float32(img)
            # mean = np.mean(img)
            # img = img - mean  # Subtract mean to center the data around zero

        if self.transform:
            if self.is_unlabeled:
                # Transform only image for unlabeled data
                augmented = self.transform(image=img)
                img = augmented['image']
            else:
                # Transform both image and mask for labeled data
                augmented = self.transform(image=img, mask=label)
                img = augmented['image']
                label = augmented['mask']

        # For unlabeled data, return only the image
        if self.is_unlabeled:
            return img
        
        return img, label

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
