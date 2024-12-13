import sys
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from src.API_functions.DL import evaluate
from src.API_functions.Images import file_batch as fb
from src.API_functions.Soils import pre_process

def get_parameters():
    return {
        'seed': 888,

        'Kfold': None,
        'ratio': 0.25,

        'model': 'U-Net',       # model = 'U-Net', 'DeepLabv3+', 'PSPNet'
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-6,
        'batch_size': 8,
        'loss_function': 'cross_entropy',

        'n_epochs': 1000,
        'patience': 50,

        'wandb': '30.no_val_transfer'
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
    #     encoder_name="resnet34",
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=1,
    # )
    model = smp.Unet(
        encoder_name="efficientnet-b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
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
    data_paths = fb.get_image_names(r'g:\DL_Data_raw\version4-classes\5.precheck_train_val\image', None, 'tif')
    labels_paths = fb.get_image_names(r'g:\DL_Data_raw\version4-classes\5.precheck_train_val\label', None, 'tif')
    
    data = fb.read_images(data_paths, 'gray', read_all=True)
    labels = fb.read_images(labels_paths, 'gray', read_all=True)
    
    for i in range(len(data)):
        data[i] = pre_process.median(data[i], kernel_size=3)
        data[i] = pre_process.histogram_equalization_float32(data[i])
    
    return data, labels
