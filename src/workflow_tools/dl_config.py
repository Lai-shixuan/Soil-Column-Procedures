import sys
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

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
        'learning_rate': 1e-4,
        'loss_function': 'cross_entropy',
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'scheduler_min_lr': 1e-6,

        'label_batch_size': 16,
        'unlabel_batch_size': 64,

        'wandb': '41.Unet-b0-half-halfLR',
        # Add semi-supervised parameters
        'consistency_weight': 0.1,
        'consistency_rampup': 100,
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
    labeled_data_paths = fb.get_image_names(r'/mnt/train_val/image', None, 'tif')
    labeled_labels_paths = fb.get_image_names(r'/mnt/train_val/label', None, 'tif')
    
    # Load unlabeled data
    unlabeled_data_paths = fb.get_image_names(r'/mnt/unlabeled', None, 'tif')
    
    labeled_data = fb.read_images(labeled_data_paths, 'gray', read_all=True)
    labels = fb.read_images(labeled_labels_paths, 'gray', read_all=True)
    unlabeled_data = fb.read_images(unlabeled_data_paths, 'gray', read_all=True)
    
    # Preprocess all data
    # for i in range(len(labeled_data)):
    #     labeled_data[i] = pre_process.median(labeled_data[i], kernel_size=3)
    #     labeled_data[i] = pre_process.histogram_equalization_float32(labeled_data[i])
    
    # for i in range(len(unlabeled_data)):
    #     unlabeled_data[i] = pre_process.median(unlabeled_data[i], kernel_size=3)
    #     unlabeled_data[i] = pre_process.histogram_equalization_float32(unlabeled_data[i])
    
    return labeled_data, labels, unlabeled_data
