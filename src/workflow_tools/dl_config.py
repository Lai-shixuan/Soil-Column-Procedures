import sys
import torch
import os
import albumentations as A
import segmentation_models_pytorch as smp
import pandas as pd

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Tuple
from src.API_functions.DL import evaluate
from src.API_functions.Images import file_batch as fb

def get_parameters():
    config_dict = {
        # Title and seed
        'wandb': '5.7 unet++eff-b0-mse0.5',
        'seed': 3407,

        # Data related parameters
        'data_resolution': 'low',   # 'low' or 'high' or 'both'
        'label_batch_size': 8,
        'ratio': 0.20,
        'Kfold': None,

        # Model related parameters
        'model': 'U-Net++',         # model = 'U-Net', 'DeepLabv3+', 'PSPNet', 'U-Net++', 'Segformer', 'UPerNet', 'Linknet'
        'encoder': 'efficientnet-b0',
        'optimizer': 'adam',        # optimizer = 'adam', 'adamw', 'sgd'
        # 'weight_decay': 0.01,     # weight_decay = 0.01
        'loss_function': 'cross_entropy',
        'transform': 'basic',

        # Learning related parameters
        'learning_rate': 1e-4,
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 20,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-6,

        # Add semi-supervised parameters
        'mode': 'semi',             # 'supervised' or 'semi'
        'unlabel_batch_size': 16,
        'consistency_weight': 0.5,
        'consistency_rampup': 30,

        # Batch debug mode and with earyly stopping
        'n_epochs': 400,
        'patience': 75,
        'batch_debug': False,

        # Scenarios, linux can compile, windows can't
        'compile': True,

        # Try to update labels, failed before
        'update': False
    }


    return config_dict

def get_debug_param_sets():
    return [
        {**get_parameters(), 'learning_rate': 3e-5, 'wandb': '4-4-v3_lr_3e-5'},
        {**get_parameters(), 'learning_rate': 1e-4, 'wandb': '4-5-v3_lr_1e-4'},
        {**get_parameters(), 'learning_rate': 6e-5, 'wandb': '4-6-v3_lr_6e-5'},
    ]

def get_transforms(seed_value):
    # Geometric transforms that affect structure
    geometric_transform = A.Compose([
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.8),
        A.RandomRotate90(p=0.8),
    ], seed=seed_value)

    # Non-geometric transforms that only affect appearance
    non_geometric_transform = A.Compose([
        A.GaussNoise(p=0.5),
        A.GaussianBlur(p=0.8, blur_limit=(3, 5)),
        # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), p=0.8),
        # A.RandomShadow(p=0.5),
        ToTensorV2(),
    ], seed=seed_value)

    # Combined transform for supervised training
    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.5),
        A.GaussianBlur(p=0.8, blur_limit=(3, 5)),
        # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), p=1),
        # A.RandomShadow(p=0.5),
        ToTensorV2(),
    ], seed=seed_value)

    transform_val = A.Compose([
        ToTensorV2(),
    ], seed=seed_value)
    
    return transform_train, transform_val, geometric_transform, non_geometric_transform

def setup_model():
    model = smp.UnetPlusPlus(
        encoder_name=get_parameters()['encoder'],
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
    # model = fr_unet.FR_UNet(num_channels=1, num_classes=1, feature_scale=2, dropout=0.2, fuse=True, out_ave=True)
    return model

def setup_training(model, learning_rate, scheduler_factor, scheduler_patience, scheduler_min_lr):
    # parameters = get_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        # weight_decay=parameters['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )
    criterion = evaluate.DiceBCELoss()
    mse_criterion = evaluate.MaskedMSELoss()
    return optimizer, scheduler, criterion, mse_criterion

def get_data_paths():
    """Define all data paths in a central location"""
    return {
        'low': {
            # 'image_dir': r'g:\DL_Data_raw\version8-low-precise\7.Final_dataset\train_val\image',
            # 'label_dir': r'g:\DL_Data_raw\version8-low-precise\7.Final_dataset\train_val\label',
            # 'padding_info': r'g:\DL_Data_raw\version8-low-precise\7.Final_dataset\train_val\image_patches.csv',
            'image_dir': r'/mnt/version8/image',
            'label_dir': r'/mnt/version8/label',
            'padding_info': r'/mnt/version8/image_patches.csv',
        },
        'high': {
            'image_dir': r'g:\DL_Data_raw\version6-large\7.Final_dataset\train_val\image',
            'label_dir': r'g:\DL_Data_raw\version6-large\7.Final_dataset\train_val\label',
            'padding_info': r'g:\DL_Data_raw\version6-large\7.Final_dataset\train_val\image_patches.csv',
        },
        'unlabeled': {
            # 'image_dir': r'g:\DL_Data_raw\version7-large-lowRH\8.Unlabeled\6.Precheck\image',
            # 'padding_info': r'g:\DL_Data_raw\version7-large-lowRH\8.Unlabeled\6.Precheck\image_patches.csv',
            'image_dir': r'/mnt/version7/unlabel/image',
            'padding_info': r'/mnt/version7/unlabel/image_patches.csv',
        }
    }

def get_image_output_paths() -> Tuple[Path, Path]:
    """For updating labels in supervised mode"""
    train_sample = Path('/root/Soil-Column-Procedures/data/noisy_reduction/1228-1/train')
    val_sample = Path('/root/Soil-Column-Procedures/data/noisy_reduction/1228-1/val')
    
    if not train_sample.exists():
        if sys.platform == 'win64':
            train_sample.mkdir(parents=True, exist_ok=True)
    if not val_sample.exists():
        if sys.platform == 'win64':
            val_sample.mkdir(parents=True, exist_ok=True)

    return train_sample, val_sample

def load_dataset(data_paths, mode='labeled'):
    """Load dataset based on provided paths"""
    if mode == 'labeled':
        images = fb.get_image_names(data_paths['image_dir'], None, 'tif')
        labels = fb.get_image_names(data_paths['label_dir'], None, 'tif')
        padding_info = pd.read_csv(data_paths['padding_info'])
        return images, labels, padding_info
    else:
        images = fb.get_image_names(data_paths['image_dir'], None, 'tif')
        padding_info = pd.read_csv(data_paths['padding_info'])
        return images, padding_info

def load_and_preprocess_data():
    params = get_parameters()
    data_paths = get_data_paths()
    labeled_data_paths, labeled_labels_paths = [], []
    padding_info = pd.DataFrame()

    # Load data based on resolution setting
    resolutions = []
    if params['data_resolution'] in ['low', 'both']:
        resolutions.append('low')
    if params['data_resolution'] in ['high', 'both']:
        resolutions.append('high')

    # Load labeled data for selected resolutions
    for res in resolutions:
        images, labels, res_padding_info = load_dataset(data_paths[res])
        labeled_data_paths.extend(images)
        labeled_labels_paths.extend(labels)
        padding_info = pd.concat([padding_info, res_padding_info], ignore_index=True)

    # Read all images at once
    labeled_data = fb.read_images(labeled_data_paths, 'gray', read_all=True)
    labels = fb.read_images(labeled_labels_paths, 'gray', read_all=True)

    # Handle unlabeled data for semi-supervised mode
    unlabeled_data = None
    unlabeled_padding_info = None
    if params['mode'] == 'semi':
        unlabeled_images, unlabeled_padding_info = load_dataset(data_paths['unlabeled'], mode='unlabeled')
        unlabeled_data = fb.read_images(unlabeled_images, 'gray', read_all=True)

    return labeled_data, labels, unlabeled_data, padding_info, unlabeled_padding_info
