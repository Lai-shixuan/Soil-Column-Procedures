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
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold, train_test_split
from src.API_functions.DL import load_data, log, seed, evaluate
from src.workflow_tools import dl_config
from src.workflow_tools.cvat_noisy import cvat_nosiy


# Global flag to track interruption
interrupted = False

# ------------------- Setup -------------------

def setup_environment(my_parameters):
    device = 'cuda'
    mylogger = log.DataLogger('wandb')

    seed.stablize_seed(my_parameters['seed'])
    torch.use_deterministic_algorithms(False)    # to ensure reproducibility
    transform_train, transform_val, geometric_transform, non_geometric_transform = dl_config.get_transforms(my_parameters['seed'])

    model = dl_config.setup_model()
    if my_parameters['compile']:
        model = torch.compile(model).to(device)
    else:
        model = model.to(device)

    # Create teacher model
    teacher_model = dl_config.setup_model()
    if my_parameters['compile']:
        teacher_model = torch.compile(teacher_model)
    teacher_model.load_state_dict(model.state_dict())

    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer, scheduler, criterion, mse_criterion = dl_config.setup_training(
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
        project="Precise-annotation",
        name=my_parameters['wandb'],
        config=my_parameters,
    )
    return model, teacher_model, device, optimizer, scheduler, criterion, mse_criterion, scaler, transform_train, transform_val, geometric_transform, non_geometric_transform, mylogger

# ------------------- Signal Handling -------------------

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

def register_signals():
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

# ------------------- Data -------------------

def prepare_data(my_parameters, transform_train, transform_val, geometric_transform):
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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=my_parameters['label_batch_size'], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=my_parameters['label_batch_size'], shuffle=False, drop_last=False)

    print(f'len of train_data: {len(train_data)}, len of val_data: {len(val_data)}')

    if my_parameters['mode'] == 'semi':
        unlabeled_dataset = load_data.my_Dataset(
            unlabeled_data, 
            [None]*len(unlabeled_data), 
            unlabeled_padding_info, 
            transform=geometric_transform  # Use only geometric transforms for unlabeled data
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=my_parameters['unlabel_batch_size'], 
            shuffle=True,
            drop_last=True
        )
        unlabeled_iter = iter(unlabeled_loader)
        print(f'len of unlabeled_data: {len(unlabeled_data)}')

    return train_dataset, val_dataset, train_loader, val_loader, unlabeled_loader if my_parameters['mode'] == 'semi' else None, unlabeled_iter if my_parameters['mode'] == 'semi' else None

# ------------------- Consistency Loss -------------------

def deal_with_nan(epoch, model_output):
    """Deal with NaN values in model output."""
    if torch.isnan(model_output).any():
    
        model_output_no_nan = torch.nan_to_num(model_output, nan=0.0)
        mean_value = model_output_no_nan.mean()
        model_output = torch.where(torch.isnan(model_output), mean_value, model_output)

        nan_count = torch.sum(torch.isnan(model_output))
        print(f"In {epoch}, Warning: {nan_count} NaN values in model_output.")

    return model_output

# Helper functions for semi-supervised mode
def fetch_unlabeled_batch(unlabeled_iter, unlabeled_loader):
    """Fetches a batch from the unlabeled data loader. If the iterator is exhausted, it resets the iterator and changes the transform to geometric."""
    try:
        batch, mask = next(unlabeled_iter)
    except StopIteration:
        unlabeled_iter = iter(unlabeled_loader)
        batch, mask = next(unlabeled_iter)
    return batch, mask, unlabeled_iter

def update_teacher_model(teacher_model, student_model, alpha=0.99):
    """Update teacher model by exponential moving average of student weights."""
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data

def compute_consistency_loss(student_model, teacher_model, device, non_geometric_transform,
                            labeled_images, unlabeled_images, labeled_masks, unlabeled_masks,
                            epoch, my_parameters, criterion):
    rampup = np.clip(epoch / my_parameters['consistency_rampup'], 0, 1)
    weight = my_parameters['consistency_weight'] * rampup

    with torch.no_grad():
        output_unlabeled = teacher_model(unlabeled_images)
        output_labeled = teacher_model(labeled_images)

    output_unlabeled = deal_with_nan(epoch, output_unlabeled)
    teacher_pred_unlabeled = torch.sigmoid(output_unlabeled)
    output_labeled = deal_with_nan(epoch, output_labeled)
    teacher_pred_labeled = torch.sigmoid(output_labeled)

    teacher_pred_labeled = teacher_pred_labeled.squeeze(1)
    teacher_pred_unlabeled = teacher_pred_unlabeled.squeeze(1)

    # Unlabeled images
    non_geometric_batch_un = []
    for img in unlabeled_images:
        # Convert tensor to numpy array in the correct format (H,W,C)
        img_np = img.cpu().permute(1, 2, 0).numpy()
        # Apply non-geometric transforms
        non_geometric = non_geometric_transform(image=img_np)['image']
        # Convert back to the correct format (C,H,W)
        non_geometric_batch_un.append(non_geometric)
    
    augmented_images = torch.stack(non_geometric_batch_un).to(device)
    student_pred_unlabeled = student_model(augmented_images).squeeze(1)
    cons_loss_un = criterion(student_pred_unlabeled, teacher_pred_unlabeled, unlabeled_masks)

    # Labeled images
    non_geometric_batch_labeled = []
    for img in labeled_images:
        # Convert tensor to numpy array in the correct format (H,W,C)
        img_np = img.cpu().permute(1, 2, 0).numpy()
        # Apply non-geometric transforms
        non_geometric = non_geometric_transform(image=img_np)['image']
        # Convert back to the correct format (C,H,W)
        non_geometric_batch_labeled.append(non_geometric)
    
    augmented_label_images = torch.stack(non_geometric_batch_labeled).to(device)
    student_pred_labeled = student_model(augmented_label_images).squeeze(1)
    cons_loss_labeled = criterion(student_pred_labeled, teacher_pred_labeled, labeled_masks)

    return weight * (cons_loss_un + cons_loss_labeled)

# ------------------- Epoch -------------------

def train_one_epoch(model, device, train_loader, my_parameters, unlabeled_loader, unlabeled_iter, non_geometric_transform, criterion, mse_criterion, optimizer, scaler, proceed_once, epoch, teacher_model):
    model.train()

    # Initialize loss variables
    supervised_total = 0.0
    soft_dice_total = 0.0
    if my_parameters['mode'] == 'semi':
        consistency_loss_total = 0.0

    for images, labels, masks in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        if my_parameters['mode'] == 'semi':
            unlabeled_images, unlabeled_masks, unlabeled_iter = fetch_unlabeled_batch(
                unlabeled_iter, unlabeled_loader)
            unlabeled_images = unlabeled_images.to(device)
            unlabeled_masks = unlabeled_masks.to(device)

        with autocast(device_type='cuda'):
            outputs = model(images)
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            supervised_loss, soft_dice = criterion(outputs, labels, masks)

            if my_parameters['mode'] == 'semi':
                cons_loss = compute_consistency_loss(
                    model, teacher_model, device, non_geometric_transform,
                    images, unlabeled_images,
                    masks, unlabeled_masks,
                    epoch, my_parameters, mse_criterion
                )
                supervised_loss = (1 - my_parameters['consistency_weight']) * supervised_loss
                total_loss = supervised_loss + cons_loss
            else:
                total_loss = supervised_loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update teacher model via EMA
        if my_parameters['mode'] == 'semi':
            update_teacher_model(teacher_model, model)

        supervised_total += supervised_loss.item()
        soft_dice_total += soft_dice.item()
        if my_parameters['mode'] == 'semi':
            consistency_loss_total += cons_loss.item()

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

    # For each epoch, divide the total loss by the number of samples
    train_loss_mean = supervised_total / len(train_loader)
    soft_dice_mean = soft_dice_total / len(train_loader)
    if my_parameters['mode'] == 'semi':
        consistency_loss_mean = consistency_loss_total / len(train_loader)
        total_loss_mean = train_loss_mean + consistency_loss_mean
    else:
        total_loss_mean = train_loss_mean
    
    return train_loss_mean, consistency_loss_mean if my_parameters['mode'] == 'semi' else None, total_loss_mean if my_parameters['mode'] == 'semi' else train_loss_mean, soft_dice_mean

def validate(model, device, val_loader, criterion):
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
            
            loss, _ = criterion(outputs, labels, masks)
            
            val_loss += loss.item()

    val_loss_mean = val_loss / len(val_loader)
    return val_loss_mean

def calculate_update(
    soft_dice_list, epoch, device, model, train_dataset, val_dataset, my_parameters):
    """Calculate update based on soft dice scores"""
    soft_dice_array = np.stack(soft_dice_list)
    train_update_path, val_update_path = dl_config.get_image_output_paths()
    update_status = cvat_nosiy.UpdateStrategy.if_update(soft_dice_array, epoch, threshold=0.9)

    if update_status:
        if my_parameters['mode'] == 'supervised':
            train_eval_loader = DataLoader(
                train_dataset,
                batch_size=my_parameters['label_batch_size'],
                shuffle=False
            )
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

            val_eval_loader = DataLoader(
                val_dataset,
                batch_size=my_parameters['label_batch_size'],
                shuffle=False
            )
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
            for idx in sample_indices:
                if idx < len(train_dataset.labels):
                    label_array = train_dataset.labels[idx]
                    cv2.imwrite(train_update_path / f'{idx}-{epoch}.tif', label_array)
                if idx < len(val_dataset.labels):
                    label_array = val_dataset.labels[idx]
                    cv2.imwrite(val_update_path / f'{idx}-{epoch}.tif', label_array)

            train_dataset.set_use_transform(True)
            val_dataset.set_use_transform(True)
        print(f"Update at epoch {epoch}")
    return update_status

def main():
    base_params = dl_config.get_parameters()
    seed.stablize_seed(base_params['seed'])
    if base_params['batch_debug']:
        debug_sets = dl_config.get_debug_param_sets()
        for debug_params in debug_sets:
            print(f"\nRunning with params: {debug_params}")
            run_experiment(debug_params)
    else:
        run_experiment(base_params)

def run_experiment(my_parameters):
    model, teacher_model, device, optimizer, scheduler, criterion, mse_criterion, scaler, transform_train, transform_val, geometric_transform, non_geometric_transform, mylogger = setup_environment(my_parameters)
    register_signals()

    train_dataset, val_dataset, train_loader, val_loader, unlabeled_loader, unlabeled_iter = prepare_data(my_parameters, transform_train, transform_val, geometric_transform)

    val_loss_best = float('inf')
    no_improvement_count = 0
    proceed_once = True
    soft_dice_list: List[float] = []

    try:
        for epoch in range(my_parameters['n_epochs']):

            print('')
            print(f"Epoch {epoch} of {my_parameters['n_epochs']}")

            # ------------------- Training -------------------

            train_loss_mean, consistency_loss_mean, total_loss_mean, soft_dice_mean = train_one_epoch(
                model, device, train_loader, my_parameters, unlabeled_loader,
                unlabeled_iter, non_geometric_transform, criterion, mse_criterion, optimizer, scaler, proceed_once, epoch,
                teacher_model
            )
            proceed_once = False
            soft_dice_list.append(soft_dice_mean)

            # ------------------- Validation -------------------

            val_loss_mean = validate(model, device, val_loader, criterion)
            current_lr = optimizer.param_groups[0]['lr']
            
            scheduler.step(val_loss_mean)

            if device == 'cuda':
                torch.cuda.empty_cache()

            # ------------------- Calculate Update -------------------

            if my_parameters['update'] == True:
                calculate_update(
                    soft_dice_list, epoch, device, model,
                    train_dataset, val_dataset, my_parameters
                )

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
                no_improvement_count = 0
                torch.save(model.state_dict(), f"src/workflow_tools/pths/model_{my_parameters['model']}_{my_parameters['wandb']}.pth")
                print(f'Model saved at epoch {epoch:.3f}, val_loss: {val_loss_mean:.3f}')
            else:
                no_improvement_count += 1                
                if no_improvement_count >= my_parameters['patience']:
                    print(f"No improvement for {my_parameters['patience']} epochs, stopping early.")
                    break

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Best validation loss: {val_loss_best}")
        wandb.finish()
        raise
    finally:
        print(f"Best validation loss: {val_loss_best}")
        wandb.finish()

if __name__ == "__main__":
    main()
