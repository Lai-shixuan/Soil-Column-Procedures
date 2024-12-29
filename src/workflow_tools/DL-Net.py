import sys
import torch
import wandb
import signal
import numpy as np
import os
import cv2

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, "/root/Soil-Column-Procedures")
# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from tqdm import tqdm
from typing import List
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold, train_test_split
from src.API_functions.DL import load_data, log, seed
from src.workflow_tools import dl_config
from src.workflow_tools.cvat_noisy import cvat_nosiy


# ------------------- Setup -------------------

my_parameters = dl_config.get_parameters()
device = 'cuda'
mylogger = log.DataLogger('wandb')

seed.stablize_seed(my_parameters['seed'])
torch.use_deterministic_algorithms(False)
transform_train, transform_val, geometric_transform, non_geometric_transform = dl_config.get_transforms(my_parameters['seed'])

model = dl_config.setup_model()
if my_parameters['compile']:
    model = torch.compile(model).to(device)
else:
    model = model.to(device)

optimizer, scheduler, criterion = dl_config.setup_training(
    model,
    my_parameters['learning_rate'],
    my_parameters['scheduler_factor'],
    my_parameters['scheduler_patience'],
    my_parameters['scheduler_min_lr']
)

# Add after device definition
scaler = GradScaler('cuda')

# Initialize wandb
wandb.init(
    project="U-Net",
    name=my_parameters['wandb'],
    config=my_parameters,
)

# ------------------- Signal Handling -------------------

# Global flag to track interruption
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    if interrupted:
        print("\nForced exit...")
        sys.exit(1)
    
    interrupted = True
    print(f"\nCaught signal {signum}. Gracefully shutting down...")
    
    try:
        # Cleanup wandb
        if wandb.run is not None:
            wandb.finish()
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    sys.exit(0)

# Register multiple signals
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request

# ------------------- Data -------------------

# Load data
labeled_data, labels, unlabeled_data, padding_info, unlabeled_padding_info = dl_config.load_and_preprocess_data()

# Split data and padding info together
train_data, val_data, train_labels, val_labels, train_padding_info, val_padding_info = train_test_split(
    labeled_data, 
    labels,
    padding_info,
    test_size=my_parameters['ratio'], 
    random_state=my_parameters['seed'],
    shuffle=False
)

# Create datasets
train_dataset = load_data.my_Dataset(train_data, train_labels, train_padding_info, transform=transform_train)
val_dataset = load_data.my_Dataset(val_data, val_labels, val_padding_info, transform=transform_val)

if my_parameters['mode'] == 'semi':
    unlabeled_dataset = load_data.my_Dataset(
        unlabeled_data, 
        [None]*len(unlabeled_data), 
        unlabeled_padding_info, 
        transform=geometric_transform  # Use only geometric transforms for unlabeled data
    )

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=my_parameters['label_batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=my_parameters['label_batch_size'], shuffle=False, drop_last=True)

if my_parameters['mode'] == 'semi':
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        batch_size=my_parameters['unlabel_batch_size'], 
        shuffle=True
    )
    unlabeled_iter = iter(unlabeled_loader)

if my_parameters['mode'] == 'semi':
    print(f'len of train_data: {len(train_data)}, len of val_data: {len(val_data)}, len of unlabeled_data: {len(unlabeled_data)}')
elif my_parameters['mode'] == 'supervised':
    print(f'len of train_data: {len(train_data)}, len of val_data: {len(val_data)}')

# ------------------- Consistency Loss -------------------

# Helper functions for semi-supervised mode
def fetch_unlabeled_batch(unlabeled_iter, unlabeled_loader):
    try:
        batch = next(unlabeled_iter)
    except StopIteration:
        # 创建新的iterator而不是重置数据
        unlabeled_loader.dataset.transform = geometric_transform  # 确保使用正确的transform
        unlabeled_iter = iter(unlabeled_loader)
        batch = next(unlabeled_iter)
    return batch, unlabeled_iter  # 只返回图像数据，因为我们不需要标签

def compute_consistency_loss(model, device, non_geometric_transform, unlabeled_images, epoch, my_parameters):
    rampup = np.clip(epoch / my_parameters['consistency_rampup'], 0, 1)
    weight = my_parameters['consistency_weight'] * rampup
    
    # For pred1, use the images directly from dataloader (already has geometric transforms)
    with torch.no_grad():
        pred1 = torch.sigmoid(model(unlabeled_images))  # As target
    
    # For pred2, only apply non-geometric transforms
    non_geometric_batch = []
    for img in unlabeled_images:
        # Convert tensor to numpy array in the correct format (H,W,C)
        img_np = img.cpu().permute(1, 2, 0).numpy()
        # Apply non-geometric transforms
        non_geometric = non_geometric_transform(image=img_np)['image']
        # Convert back to the correct format (C,H,W)
        non_geometric_batch.append(non_geometric)
    
    augmented_images = torch.stack(non_geometric_batch).to(device)
    
    pred2 = model(augmented_images)
    
    cons_loss, _ = criterion(pred2, pred1) # pred1 is target
    return weight * cons_loss

# ------------------- Epoch -------------------

val_loss_best = float('inf')
proceed_once = True
soft_dice_list: List[float] = []

train_sample = Path('/root/Soil-Column-Procedures/data/noisy_reduction/1228-1/train')
val_sample = Path('/root/Soil-Column-Procedures/data/noisy_reduction/1228-1/val')

if not train_sample.exists():
    train_sample.mkdir(parents=True)
if not val_sample.exists():
    val_sample.mkdir(parents=True)

try:
    for epoch in range(my_parameters['n_epochs']):

        print('')
        print(f"Epoch {epoch} of {my_parameters['n_epochs']}")

        # ------------------- Training -------------------

        model.train()

        # Initialize loss variables
        train_loss = 0.0
        if my_parameters['mode'] == 'semi':
            consistency_loss = 0.0

        for batch_idx, (images, labels, masks) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            if my_parameters['mode'] == 'semi':
                unlabeled_images, unlabeled_iter = fetch_unlabeled_batch(
                    unlabeled_iter, unlabeled_loader
                )
                unlabeled_images = unlabeled_images.to(device)

            with autocast(device_type='cuda'):
                outputs = model(images)
                if outputs.dim() == 4 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)
                
                supervised_loss, soft_dice = criterion(outputs, labels, masks)

                if my_parameters['mode'] == 'semi':
                    cons_loss = compute_consistency_loss(
                        model, device, non_geometric_transform,
                        unlabeled_images, epoch, my_parameters
                    )
                    total_loss = supervised_loss + cons_loss
                else:
                    total_loss = supervised_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += supervised_loss.item() * images.size(0)
            if my_parameters['mode'] == 'semi':
                consistency_loss += cons_loss.item() * unlabeled_images.size(0)

            # In the first iteration, print some information
            if proceed_once:
                print(f'outputs.size(): {outputs.size()}, labels.size(): {labels.size()}')
                print(f'outputs.min: {outputs.min()}, outputs.max: {outputs.max()}')
                print(f'images.min: {images.min()}, images.max: {images.max()}')
                print(f'labels.min: {labels.min()}, labels.max: {labels.max()}')
                print(f'count of label 0: {(labels == 0).sum()}, count of label 1:{(labels == 1).sum()}')
                if my_parameters['mode'] == 'semi':
                    print(f"consistency loss: {cons_loss.item()}, weight: {my_parameters['consistency_weight'] * np.clip(epoch / my_parameters['consistency_rampup'], 0, 1)}")
                print('')
                proceed_once = False

        # For each epoch, divide the total loss by the number of samples
        train_loss_mean = train_loss / (len(train_loader) * train_loader.batch_size)
        soft_dice_mean = soft_dice / (len(train_loader) * train_loader.batch_size)
        if my_parameters['mode'] == 'semi':
            consistency_loss_mean = consistency_loss / (len(train_loader) * unlabeled_loader.batch_size)
            total_loss_mean = train_loss_mean + consistency_loss_mean
        else:
            total_loss_mean = train_loss_mean
        
        soft_dice_list.append(soft_dice_mean.cpu().item())

        # ------------------- Validation -------------------

        model.eval()
        val_loss = 0

        # Update validation loop autocast
        with torch.no_grad(), autocast(device_type='cuda'):
            for images, labels, masks in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                if outputs.dim() == 4 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)                

                loss, _ = criterion(outputs, labels, masks)
                
                val_loss += loss.item() * images.size(0)

        val_loss_mean = val_loss / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step the scheduler
        scheduler.step(val_loss_mean)

        # Clear cache after each epoch
        if device == 'cuda':
            torch.cuda.empty_cache()

        # ------------------- Calculate Update -------------------

        soft_dice_array = np.stack(soft_dice_list)
        update_status = cvat_nosiy.UpdateStrategy.if_update(soft_dice_array, epoch, threshold=0.9)
        if False and update_status:

            # ------------------- Label Refinement -------------------

            if my_parameters['mode'] == 'supervised':
                # Small-batch label refinement
                train_eval_loader = DataLoader(train_dataset, batch_size=my_parameters['label_batch_size'], shuffle=False)
                train_dataset.set_use_transform(False)
                for batch_idx, (imgs, lbls, msks) in enumerate(tqdm(train_eval_loader)):
                    imgs = imgs.to(device)
                    imgs = imgs.unsqueeze(1)
                    with torch.no_grad():
                        preds = torch.sigmoid(model(imgs))
                        preds = preds.squeeze(1)
                    for i in range(len(preds)):
                        dataset_idx = batch_idx * train_eval_loader.batch_size + i
                        train_dataset.update_label_by_index(dataset_idx, preds[i], threshold=0.8)

                val_eval_loader = DataLoader(val_dataset, batch_size=my_parameters['label_batch_size'], shuffle=False)
                val_dataset.set_use_transform(False)
                for batch_idx, (imgs, lbls, msks) in enumerate(tqdm(val_eval_loader)):
                    imgs = imgs.to(device)
                    imgs = imgs.unsqueeze(1)
                    with torch.no_grad():
                        preds = torch.sigmoid(model(imgs))
                        preds = preds.squeeze(1)
                    for i in range(len(preds)):
                        dataset_idx = batch_idx * val_eval_loader.batch_size + i
                        val_dataset.update_label_by_index(dataset_idx, preds[i], threshold=0.8)

                # Print label stats for selected indices
                sample_indices = range(0, 101, 10)
                # stats_train, stats_val = {}, {}
                for idx in sample_indices:
                    if idx < len(train_dataset.labels):
                        label_array = train_dataset.labels[idx]
                        cv2.imwrite(train_sample / f'{idx}-{epoch}.tif', label_array)
                        # stats_train[idx] = (np.sum(label_array == 0), np.sum(label_array == 1))
                    if idx < len(val_dataset.labels):
                        label_array = val_dataset.labels[idx]
                        cv2.imwrite(val_sample / f'{idx}-{epoch}.tif', label_array)
                        # stats_val[idx] = (np.sum(label_array == 0), np.sum(label_array == 1))
                # print(f"Train label stats after update: {stats_train}")
                # print(f"Val label stats after update: {stats_val}")

            train_dataset.set_use_transform(True) 
            val_dataset.set_use_transform(True)
            print(f"Update at epoch {epoch}")

        # ------------------- Logging -------------------

        if my_parameters['mode'] == 'semi':
            dict_to_log = {
                'epoch': epoch,
                'supervised_loss': train_loss_mean,
                'cons_loss': consistency_loss_mean,
                'total_loss': total_loss_mean,
                'val_loss': val_loss_mean,
                'learning_rate': current_lr
            }
        else:
            dict_to_log = {
                'epoch': epoch,
                'total_loss': total_loss_mean,
                'val_loss': val_loss_mean,
                'learning_rate': current_lr
            }

        mylogger.log(dict_to_log)

        if val_loss_mean < val_loss_best:
            val_loss_best = val_loss_mean
            torch.save(model.state_dict(), f"src/workflow_tools/pths/model_{my_parameters['model']}_{my_parameters['wandb']}.pth")
            print(f'Model saved at epoch {epoch:.3f}, val_loss: {val_loss_mean:.3f}')

except Exception as e:
    print(f"An error occurred: {e}")
    wandb.finish()
    raise
finally:
    wandb.finish()
