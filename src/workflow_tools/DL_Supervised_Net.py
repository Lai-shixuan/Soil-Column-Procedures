import sys
import torch
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")
# sys.path.insert(0, "/root/Soil-Column-Procedures")

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # Add this import
from sklearn.model_selection import KFold, train_test_split
from src.API_functions.DL import load_data, log, seed 
from src.workflow_tools import dl_config

import wandb

# ------------------- Setup -------------------

my_parameters = dl_config.get_parameters()
device = 'cuda'
mylogger = log.DataLogger('wandb')  # 'wandb' or 'all'

seed.stablize_seed(my_parameters['seed'])
transform_train, transform_val = dl_config.get_transforms(my_parameters['seed'])
model = dl_config.setup_model()
model = model.to(device)

# model.load_state_dict(torch.load(f"c:/Users/laish/1_Codes/Image_processing_toolchain/src/workflow_tools/model_DeepLabv3+_23.drive_again.pth", weights_only=True))
# Freeze encoder parameters
# for param in model.encoder.parameters():
    # param.requires_grad = False

# Add after device definition
scaler = GradScaler()

optimizer, scheduler, criterion = dl_config.setup_training(
    model,
    my_parameters['learning_rate'],
    my_parameters['scheduler_factor'],
    my_parameters['scheduler_patience'],
    my_parameters['scheduler_min_lr']
)

wandb.init(
    project="U-Net",
    name=my_parameters['wandb'],
    config=my_parameters,
)

# ------------------- Data Loading -------------------

# Load data
labeled_data, labels, _, padding_info, _ = dl_config.load_and_preprocess_data()

# Split data and padding info together
train_data, val_data, train_labels, val_labels, train_padding_info, val_padding_info = train_test_split(
    labeled_data, 
    labels,
    padding_info,
    test_size=my_parameters['ratio'], 
    random_state=my_parameters['seed']
)

# Create datasets with padding info
train_dataset = load_data.my_Dataset(train_data, train_labels, train_padding_info, transform=transform_train)
val_dataset = load_data.my_Dataset(val_data, val_labels, val_padding_info, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=my_parameters['label_batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=my_parameters['label_batch_size'], shuffle=False)

print(f'len of train_data: {len(train_data)}, len of val_data: {len(val_data)}')

# ------------------- Training -------------------

val_loss_best = float('inf')
proceed_once = True  # Add a flag

for epoch in range(my_parameters['n_epochs']):
    model.train()
    train_loss = 0.0

    for images, labels, masks in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        # Enable autocasting for forward pass
        with autocast(device_type='cuda'):
            outputs = model(images)

            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            
            # Only proceed once:
            if proceed_once:
                print(f'outputs.size(): {outputs.size()}, labels.size(): {labels.size()}')
                print(f'outputs.min: {outputs.min()}, outputs.max: {outputs.max()}')
                print(f'images.min: {images.min()}, images.max: {images.max()}')
                print(f'labels.min: {labels.min()}, labels.max: {labels.max()}')
                print(f'count of label 0: {(labels == 0).sum()}, count of label 1:{(labels == 1).sum()}')
                print('')
                proceed_once = False  # Set the flag to False after proceeding once
            
            loss = criterion(outputs, labels, masks)
        
        # Modified backward and optimize with scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * images.size(0)
    
    train_loss_mean = train_loss / len(train_loader.dataset)

    # ------------------- Validation -------------------

    model.eval()
    val_loss = 0

    # Update validation loop with autocast
    with torch.no_grad(), autocast(device_type='cuda'):
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
                
            loss = criterion(outputs, labels, masks)
            val_loss += loss.item() * images.size(0)

    val_loss_mean = val_loss / len(val_loader.dataset)
    current_lr = optimizer.param_groups[0]['lr']
    dict = {
        'train_loss': train_loss_mean,
        'epoch': epoch,
        'val_loss': val_loss_mean,
        'learning_rate': current_lr
    }
    mylogger.log(dict)

    # Step the scheduler
    scheduler.step(val_loss_mean)

    if val_loss_mean < val_loss_best:
        val_loss_best = val_loss_mean
        torch.save(model.state_dict(), f"model_{my_parameters['model']}_{my_parameters['wandb']}.pth")
        print(f'Model saved at epoch {epoch:.3f}, val_loss: {val_loss_mean:.3f}')

wandb.finish()