import sys
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import pandas as pd

sys.path.insert(0, "/root/Soil-Column-Procedures")
# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from src.API_functions.DL import evaluate
from src.API_functions.Images import file_batch as fb

def get_parameters():
    return {
        'seed': 3407,

        'Kfold': None,
        'ratio': 0.25,
        'n_epochs': 1000,
        'patience': 50,

        'model': 'U-Net',       # model = 'U-Net', 'DeepLabv3+', 'PSPNet'
        'encoder': 'efficientnet-b0',
        'optimizer': 'adam',
        'learning_rate': 5e-5,
        'loss_function': 'cross_entropy',
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-6,

        'label_batch_size': 16,

        'wandb': '49.test',

        # Add semi-supervised parameters
        'unlabel_batch_size': 64,
        'consistency_weight': 0.1,
        'consistency_rampup': 100,

        'mode': 'supervised',  # 'supervised' or 'semi'
    }

def get_transforms(seed_value):
    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.GaussNoise(p=0.5),
        ToTensorV2(),
    ], seed=seed_value)

    transform_val = A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Rotate(limit=90, p=0.5),
        # A.GaussNoise(p=0.5),
        ToTensorV2(),
    ], seed=seed_value)
    
    return transform_train, transform_val

def setup_model():
    # model = smp.DeepLabV3Plus(
    #     encoder_name="efficientnet-b0",
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=1,
    # )
    model = smp.Unet(
        encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    # model = fr_unet.FR_UNet(num_channels=1, num_classes=1, feature_scale=2, dropout=0.2, fuse=True, out_ave=True)
    return model

def setup_training(model, learning_rate, scheduler_factor, scheduler_patience, scheduler_min_lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )
    criterion = evaluate.DiceBCELoss()
    return optimizer, scheduler, criterion

def load_and_preprocess_data():
    # Load labeled data
    labeled_data_paths = fb.get_image_names(r'g:\DL_Data_raw\version7-large-lowRH\7.Final_dataset\train_val\image', None, 'tif')
    labeled_labels_paths = fb.get_image_names(r'g:\DL_Data_raw\version7-large-lowRH\7.Final_dataset\train_val\label', None, 'tif')
    padding_info = pd.read_csv(r'g:\DL_Data_raw\version7-large-lowRH\7.Final_dataset\train_val\image_patches.csv')
    
    labeled_data = fb.read_images(labeled_data_paths, 'gray', read_all=True)
    labels = fb.read_images(labeled_labels_paths, 'gray', read_all=True)
    
    # Check mode and load unlabeled data only if in semi-supervised mode
    params = get_parameters()
    if params['mode'] == 'semi':
        unlabeled_data_paths = fb.get_image_names(r'/mnt/unlabeled', None, 'tif')
        unlabeled_padding_info = pd.read_csv('/mnt/unlabel_image_patches.csv')
        unlabeled_data = fb.read_images(unlabeled_data_paths, 'gray', read_all=True)
    elif params['mode'] == 'supervised':
        unlabeled_data = None
        unlabeled_padding_info = None
    else:
        raise ValueError(f"Invalid mode: {params['mode']}")
    
    return labeled_data, labels, unlabeled_data, padding_info, unlabeled_padding_info
