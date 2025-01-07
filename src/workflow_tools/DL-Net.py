import sys
import torch
import wandb
import signal
import numpy as np
import os
import cv2
from math import exp

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold, train_test_split
from src.API_functions.DL import load_data, log, seed
from src.workflow_tools import dl_config
from src.workflow_tools.cvat_noisy import cvat_nosiy
from src.workflow_tools.database import s4augmented_labels


# Global flag to track interruption
interrupted = False


# ------------------- Setup -------------------

def setup_environment(my_parameters):

    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mylogger = log.DataLogger('wandb')

    seed.stablize_seed(my_parameters['seed'])
    transform_train, transform_val, geometric_transform, non_geometric_transform = dl_config.get_transforms(my_parameters['seed'])

    model = dl_config.setup_model(my_parameters['encoder'])
    if my_parameters['compile']:
        model = torch.compile(model).to(device)
    else:
        model = model.to(device)

    # Create teacher model
    teacher_model = dl_config.setup_model(my_parameters['encoder'])
    if my_parameters['compile']:
        teacher_model = torch.compile(teacher_model)
        teacher_model.load_state_dict(model.state_dict())
        # state_dict = torch.load('data/pths/precise/model_U-Net++_16.1-supervised-scse-batch3.pth', map_location='cuda')
        # teacher_model.load_state_dict(state_dict)
        # del state_dict

    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer, scheduler, criterion, kl_criterion = dl_config.setup_training(
        model,
        my_parameters['learning_rate'],
        my_parameters['scheduler_factor'],
        my_parameters['scheduler_patience'],
        my_parameters['scheduler_min_lr'],
        my_parameters['T_max']
    )

    # Add after device definition
    scaler = GradScaler('cuda')

    # Initialize wandb
    wandb.init(
        project="Precise-annotation",
        name=my_parameters['wandb'],
        config=my_parameters,
    )
    if my_parameters['mode'] == 'semi':
        wandb.define_metric('epoch', summary='max')
        wandb.define_metric('supervised_loss', summary='min')
        wandb.define_metric('cons_loss_un', summary='min')
        wandb.define_metric('cons_loss_labeled', summary='min')
        wandb.define_metric('total_loss', summary='min')
        wandb.define_metric('val_loss', summary='min')
    else:
        wandb.define_metric('epoch', summary='max')
        wandb.define_metric('total_loss', summary='min')
        wandb.define_metric('val_loss', summary='min')

    return model, teacher_model, device, optimizer, scheduler, criterion, kl_criterion, scaler, transform_train, transform_val, geometric_transform, non_geometric_transform, mylogger

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
            transform=None  # Use only geometric transforms for unlabeled data
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

def fetch_unlabeled_batch(unlabeled_iter, unlabeled_loader):
    """Fetches a batch from the unlabeled data loader. If the iterator is exhausted, it resets the iterator and changes the transform to geometric."""
    try:
        batch, mask = next(unlabeled_iter)
    except StopIteration:
        unlabeled_iter = iter(unlabeled_loader)
        batch, mask = next(unlabeled_iter)
    return batch, mask, unlabeled_iter

def update_teacher_model(teacher_model, student_model, alpha):
    """Update teacher model by exponential moving average of student weights."""
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data

def update_ema_variables(ema_model, model, alpha):
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()
        for entry in ema_model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()
            new_param = (ema_param * alpha) + (param * (1. - alpha))
            ema_model_state_dict[entry] = new_param
        ema_model.load_state_dict(ema_model_state_dict)

def compute_consistency_loss(student_model, teacher_model, device, transform_train,
                            images, masks,
                            epoch, rampup_weight, criterion, threshold=0.8):

    with torch.no_grad():
        output = teacher_model(images)

    output = deal_with_nan(epoch, output)
    teacher_pred = torch.sigmoid(output).squeeze(1)

    threshold = threshold * rampup_weight
    confs = torch.where((teacher_pred > threshold) | (teacher_pred < 1 - threshold), 1, 0).float()

    teacher_pred = (teacher_pred > 0.5).float()

    batch_imgs = []
    batch_labels = []
    batch_masks = []
    batch_conf = []
    for img, label, mask, conf in zip(images, teacher_pred, masks, confs):
        img_np = img.squeeze(0).cpu().numpy()
        label_np = label.cpu().numpy()
        mask_np = mask.cpu().numpy()
        conf_np = conf.cpu().numpy()

        augmenter = s4augmented_labels.ImageAugmenter(img_np, label_np, additional_img=conf_np, mask=mask_np)
        augmented_img, augmented_label, augmented_conf = augmenter.augment()

        augmented = transform_train(image=augmented_img, masks=[augmented_label, mask_np, augmented_conf])
        batch_imgs.append(augmented['image'])
        batch_labels.append(augmented['masks'][0])
        batch_masks.append(augmented['masks'][1])
        batch_conf.append(augmented['masks'][2])

    trans_imgs = torch.stack(batch_imgs).to(device)
    trans_lbls = torch.stack(batch_labels).to(device)
    trans_masks = torch.stack(batch_masks).to(device)
    trans_conf = torch.stack(batch_conf).to(device)

    trans_masks = trans_conf * trans_masks

    student_pred = student_model(trans_imgs).squeeze(1)
    loss = criterion(student_pred, trans_lbls, trans_masks)
    
    return loss * rampup_weight

# ------------------- Epoch -------------------

def train_one_epoch(model, device, train_loader, my_parameters, unlabeled_loader, unlabeled_iter, transform_train, non_geometric_transform, criterion, kl_criterion, optimizer, scaler, proceed_once, epoch, teacher_model, model_good_epoch):
    model.train()

    # Initialize loss variables
    supervised_total = 0.0
    soft_dice_total = 0.0
    if my_parameters['mode'] == 'semi':
        total_cons_loss_un = 0.0
        total_cons_loss_labeled = 0.0
        total_loss_total = 0.0
        alpha = 0

    for images, labels, masks in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        if my_parameters['mode'] == 'semi':
            unlabeled_images, unlabeled_masks, unlabeled_iter = fetch_unlabeled_batch(
                unlabeled_iter, unlabeled_loader)
            unlabeled_images = unlabeled_images.to(device)
            unlabeled_masks = unlabeled_masks.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            supervised_loss = criterion(outputs, labels, masks)

            if my_parameters['mode'] == 'semi':
                rampup = my_parameters['consistency_rampup']
                # rampup = np.clip(epoch / ramup, 0, 1)
                rampup_weight = exp(-5 * (1 - epoch / rampup) ** 2)
                if epoch > rampup:
                    rampup_weight = 1
                cons_loss_un = compute_consistency_loss(
                    model, teacher_model, device, transform_train,
                    unlabeled_images, unlabeled_masks,
                    epoch, rampup_weight, criterion 
                )
                cons_loss = cons_loss_un
                # cons_loss_labeled = compute_consistency_loss(
                #     model, teacher_model, device, non_geometric_transform,
                #     images, masks,
                #     epoch, rampup, criterion 
                # )
                # un_weight = my_parameters['unlabel_batch_size'] / (my_parameters['unlabel_batch_size'] + my_parameters['label_batch_size'])
                # labeled_weight = my_parameters['label_batch_size'] / (my_parameters['unlabel_batch_size'] + my_parameters['label_batch_size'])
                # cons_loss = cons_loss_un * un_weight + cons_loss_labeled * labeled_weight

            cons_combine_weight = my_parameters['consistency_weight'] * rampup_weight
            total_loss = supervised_loss * (1 - cons_combine_weight) + cons_loss * cons_combine_weight if my_parameters['mode'] == 'semi' else supervised_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update teacher model via EMA
        if my_parameters['mode'] == 'semi':
            if epoch < 120:
                teacher_model.load_state_dict(model.state_dict())
            elif epoch <= 200:
                alpha = 0.99
                update_ema_variables(teacher_model, model, alpha=alpha)
            elif epoch > 200:
                alpha = 0.999
                update_ema_variables(teacher_model, model, alpha=alpha)

        supervised_total += supervised_loss.item()
        # soft_dice_total += soft_dice.item()
        if my_parameters['mode'] == 'semi':
            total_cons_loss_un += cons_loss_un.item()
            # total_cons_loss_labeled += cons_loss_labeled.item()
            total_loss_total += total_loss.item()

        # In the first iteration, print some information
        if proceed_once:
            print(f'outputs.size(): {outputs.size()}, labels.size(): {labels.size()}')
            print(f'outputs.min: {outputs.min()}, outputs.max: {outputs.max()}')
            print(f'images.min: {images.min()}, images.max: {images.max()}')
            print(f'labels.min: {labels.min()}, labels.max: {labels.max()}')
            print(f'count of label 0: {(labels == 0).sum()}, count of label 1:{(labels == 1).sum()}')
            if my_parameters['mode'] == 'semi':
                print(f"consistency loss: {cons_loss.item()}, weight: {my_parameters['consistency_weight'] * np.clip(epoch / my_parameters['consistency_rampup'], 0, 1)}")

    # For each epoch, divide the total loss by the number of samples
    train_loss_mean = supervised_total / len(train_loader)
    soft_dice_mean = soft_dice_total / len(train_loader)
    if my_parameters['mode'] == 'semi':
        total_cons_loss_un_m = total_cons_loss_un / len(train_loader)
        # total_cons_loss_labeled_m = total_cons_loss_labeled / len(train_loader)
        total_loss_mean = total_loss_total / len(train_loader)
    else:
        total_loss_mean = train_loss_mean

    if my_parameters['mode'] == 'semi':
        # None stands for total_cons_loss_labeled_m
        return train_loss_mean, total_cons_loss_un_m, None, total_loss_mean, soft_dice_mean, alpha
    else:
        return train_loss_mean, None, None, train_loss_mean, soft_dice_mean, None

def validate(model, device, val_loader, criterion):
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
            
            loss = criterion(outputs, labels, masks)
            
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
    model, teacher_model, device, optimizer, scheduler, criterion, kl_criterion, scaler, transform_train, transform_val, geometric_transform, non_geometric_transform, mylogger = setup_environment(my_parameters)
    register_signals()

    train_dataset, val_dataset, train_loader, val_loader, unlabeled_loader, unlabeled_iter = prepare_data(my_parameters, transform_train, transform_val, geometric_transform)

    train_loss_best = float('inf')
    val_loss_best = float('inf')
    val_teacher_loss_best = float('inf')
    no_improvement_count = 0
    proceed_once = True
    soft_dice_list: List[float] = []
    model_good_epoch = 100000

    try:
        for epoch in range(my_parameters['n_epochs']):

            print(f"Epoch {epoch} of {my_parameters['n_epochs']}")

            # ------------------- Training -------------------

            train_loss_m, cons_loss_un_m, _, total_loss_m, soft_dice_m, alpha = train_one_epoch(
                model, device, train_loader, my_parameters, unlabeled_loader,
                unlabeled_iter, transform_train, non_geometric_transform, criterion, kl_criterion, optimizer, scaler, proceed_once, epoch,
                teacher_model, model_good_epoch 
            )
            proceed_once = False
            soft_dice_list.append(soft_dice_m)

            # ------------------- Validation -------------------

            val_loss_mean = validate(model, device, val_loader, criterion)
            val_teacher_loss_mean = validate(teacher_model, device, val_loader, criterion)

            # ------------------- Scheduler -------------------
            
            current_lr = optimizer.param_groups[0]['lr']

            if my_parameters['scheduler_type'] == 'plateau':
                scheduler.step(val_loss_mean)
            elif my_parameters['scheduler_type'] == 'cosine':
                scheduler.step()

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
                    'supervised_loss': train_loss_m,
                    'cons_loss_un': cons_loss_un_m,
                    # 'cons_loss_labeled': cons_loss_labled_m,
                    'total_loss': total_loss_m,
                    'val_loss': val_loss_mean,
                    'val_teacher_loss': val_teacher_loss_mean,
                    'learning_rate': current_lr,
                    'alpha': alpha
                }
            else:
                dict_to_log = {
                    'epoch': epoch,
                    'total_loss': total_loss_m,
                    'val_loss': val_loss_mean,
                    'learning_rate': current_lr
                }

            mylogger.log(dict_to_log)

            # Log the best training, teacher val and student val loss, save the model if it is the best
            if train_loss_m < train_loss_best:
                train_loss_best = train_loss_m
                if train_loss_best < 0.20 and val_teacher_loss_best < 0.16 and val_loss_best < 0.16 and model_good_epoch == 100000:
                    model_good_epoch = epoch
                    print(f"Model is good at epoch {model_good_epoch}, now start to update teacher model.")
                print(f'New best training loss: {train_loss_best:.3f}')
            
            if val_teacher_loss_mean < val_teacher_loss_best:
                val_teacher_loss_best = val_teacher_loss_mean
                print(f'New best teacher validation loss: {val_teacher_loss_best:.3f}')

            if val_loss_mean < val_loss_best:
                val_loss_best = val_loss_mean
                no_improvement_count = 0

                path = f"data/pths/precise/model_{my_parameters['model']}_{my_parameters['wandb']}.pth"
                if not Path(path).parent.exists():
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)

                print(f'Model saved at epoch {epoch:.3f}, val_loss: {val_loss_mean:.3f}')
            else:
                no_improvement_count += 1                
                if no_improvement_count >= my_parameters['patience'] or (epoch > 300 and val_loss_mean > 0.4):
                    print(f"No improvement for {my_parameters['patience']} epochs or val_loss > 0.5, stopping training.")
                    break

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"The best validation loss was: {val_loss_best}")
        wandb.finish()
        raise
    finally:
        print(f"The best validation loss was: {val_loss_best}")
        wandb.finish()

if __name__ == "__main__":
    main()
