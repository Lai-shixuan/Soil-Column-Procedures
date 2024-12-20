import sys
import torch
from torch.amp import autocast, GradScaler  # Updated import
import wandb
import signal
import numpy as np
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, "/root/Soil-Column-Procedures")
# sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from src.API_functions.DL import load_data, log, seed
from src.workflow_tools import dl_config
from src.workflow_tools.model_online import fr_unet

# ------------------- Setup -------------------

my_parameters = dl_config.get_parameters()
device = 'cuda'
mylogger = log.DataLogger('wandb')

seed.stablize_seed(my_parameters['seed'])
transform_train, transform_val = dl_config.get_transforms(my_parameters['seed'])
model = dl_config.setup_model()
model = torch.compile(model).to(device)
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
    random_state=my_parameters['seed']
)

# Create datasets
train_dataset = load_data.my_Dataset(train_data, train_labels, train_padding_info, transform=transform_train)
val_dataset = load_data.my_Dataset(val_data, val_labels, val_padding_info, transform=transform_val)
unlabeled_dataset = load_data.my_Dataset(unlabeled_data, [None]*len(unlabeled_data), unlabeled_padding_info, transform=transform_train)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=my_parameters['label_batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=my_parameters['label_batch_size'], shuffle=False, drop_last=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=my_parameters['unlabel_batch_size'], shuffle=True)

print(f'len of train_data: {len(train_data)}, len of val_data: {len(val_data)}, len of unlabeled_data: {len(unlabeled_data)}')

def consistency_weight(epoch):
    return my_parameters['consistency_weight'] * np.clip(epoch/my_parameters['consistency_rampup'], 0, 1)

# ------------------- Training -------------------

val_loss_best = float('inf')
proceed_once = True
unlabeled_iter = iter(unlabeled_loader)

try:
    for epoch in range(my_parameters['n_epochs']):
        model.train()
        train_loss = 0.0
        consistency_loss = 0.0

        for images, labels, masks in tqdm(train_loader):
            # Get a batch of unlabeled data
            try:
                unlabeled_images = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_images = next(unlabeled_iter)

            # Handle labeled data
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            unlabeled_images = unlabeled_images.to(device)
            
            # Enable autocasting for forward pass with device type
            with autocast(device_type='cuda'):
                outputs = model(images)
                if outputs.dim() == 4 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)
                
                # Apply mask to loss calculation
                supervised_loss = criterion(outputs, labels, masks)

                # Handle unlabeled data
                with torch.no_grad():
                    pred1 = model(unlabeled_images)
                
                # Convert tensor to numpy array with correct format (BHWC)
                unlabeled_np = unlabeled_images.cpu().permute(0, 2, 3, 1).numpy()
                
                # Apply augmentation to each image in the batch
                augmented_batch = []
                for img in unlabeled_np:
                    augmented = transform_train(image=img)['image']
                    augmented_batch.append(augmented)
                
                # Stack and convert back to tensor in BCHW format
                augmented_images = torch.from_numpy(np.stack(augmented_batch)).to(device)
                pred2 = model(augmented_images)
                
                # Consistency loss
                cons_loss = torch.mean((pred1 - pred2) ** 2)
                weight = consistency_weight(epoch)
                total_loss = supervised_loss + weight * cons_loss

            # Optimize with gradient scaling
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += supervised_loss.item() * images.size(0)
            consistency_loss += cons_loss.item() * unlabeled_images.size(0)

            # Print progress info once
            if proceed_once:
                print(f'outputs.size(): {outputs.size()}, labels.size(): {labels.size()}')
                print(f'outputs.min: {outputs.min()}, outputs.max: {outputs.max()}')
                print(f'images.min: {images.min()}, images.max: {images.max()}')
                print(f'labels.min: {labels.min()}, labels.max: {labels.max()}')
                print(f'count of label 0: {(labels == 0).sum()}, count of label 1:{(labels == 1).sum()}')
                print(f'consistency loss: {cons_loss.item()}, weight: {weight}')
                print('')
                proceed_once = False

        train_loss_mean = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0

        # Update validation loop autocast
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
            torch.save(model.state_dict(), f"src/workflow_tools/pths/model_{my_parameters['model']}_{my_parameters['wandb']}.pth")
            print(f'Model saved at epoch {epoch:.3f}, val_loss: {val_loss_mean:.3f}')

except Exception as e:
    print(f"An error occurred: {e}")
    wandb.finish()
    raise
finally:
    wandb.finish()
