import sys
import torch
import torch.nn as nn
import torch.utils.data.sampler
import wandb
import signal
import pandas as pd
import os
from math import exp

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "/home/shixuan/Soil-Column-Procedures/")

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold, train_test_split
from src.API_functions.DL import load_data, log, seed
from src.workflow_tools import dl_config
from src.API_functions.Images.file_batch import windows_adjustment_one_image

# from src.workflow_tools.cvat_noisy import cvat_nosiy
# from src.workflow_tools.database import s4augmented_labels


class TrainingContext:
    """Encapsulates all components and configurations needed for training"""
    def __init__(self,
                model: Optional[torch.nn.Module] = None,
                teacher_model: Optional[torch.nn.Module] = None,
                device: Optional[torch.device] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                criterion: Optional[torch.nn.Module] = None,
                kl_criterion: Optional[torch.nn.Module] = None,
                scaler: Optional[GradScaler] = None,
                transform_train: Optional[Any] = None,
                transform_val: Optional[Any] = None,
                train_loader: Optional[DataLoader] = None,
                val_loader: Optional[DataLoader] = None,
                my_parameters: Optional[Dict[str, Any]] = None,
                logger: Optional[log.DataLogger] = None):
        self.model = model
        self.teacher_model = teacher_model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.kl_criterion = kl_criterion
        self.scaler = scaler
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.my_parameters = my_parameters or {}
        self.logger = logger


# Global flag to track interruption
interrupted = False


# ------------------- BN IN LN -------------------

def remove_bn_layers(model):
    """
    Recursively removes all BatchNorm1d and BatchNorm2d layers from a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to modify.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # Replace the BatchNorm layer with Identity
            setattr(model, name, nn.Identity())
        else:
            # Recursively call remove_bn_layers on the child module
            remove_bn_layers(module)

def replace_bn_with_in(model, exclude_modules=None):
    """
    递归地将PyTorch模型中的所有BatchNorm1d和BatchNorm2d层替换为InstanceNorm1d和InstanceNorm2d层，
    但排除指定的模块。

    Args:
        model (torch.nn.Module): 需要修改的PyTorch模型。
        exclude_modules (list of str, optional): 模块名称列表，这些模块中的BN层将不会被替换。
    """
    if exclude_modules is None:
        exclude_modules = []

    for name, module in model.named_children():
        module_path = name

        # 检查是否在排除列表中
        if any(module_path.startswith(exclude) for exclude in exclude_modules):
            # 如果当前模块在排除列表中，跳过替换，并继续遍历其子模块
            # replace_bn_with_in(module, exclude_modules)
            continue

        if isinstance(module, nn.BatchNorm1d):
            num_features = module.num_features
            in_layer = nn.InstanceNorm1d(num_features, affine=module.affine, track_running_stats=False)
            setattr(model, name, in_layer)
        elif isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            in_layer = nn.InstanceNorm2d(num_features, affine=module.affine, track_running_stats=False)
            setattr(model, name, in_layer)
        else:
            # 递归调用以处理子模块
            replace_bn_with_in(module, exclude_modules)

def count_norm_layers(model):
    bn = 0
    in_layer = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            bn += 1
        if isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):
            in_layer += 1
    return bn, in_layer


# ------------------- Environment -------------------

def setup_environment(my_parameters):

    gpu_id = my_parameters['gpu_id']
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mylogger = log.DataLogger('wandb')

    seed.stablize_seed(my_parameters['seed'])
    transform_train, transform_val, geometric_transform, non_geometric_transform = dl_config.get_transforms(my_parameters['seed'])

    model = dl_config.setup_model(my_parameters['model'], my_parameters['encoder'])
    if my_parameters['normalization'] == 'remove':
        remove_bn_layers(model)
    elif my_parameters['normalization'] == 'in':
        replace_bn_with_in(model, exclude_modules=['encoder'])
        bn_count, in_count = count_norm_layers(model)
        print(f'BatchNorm layers: {bn_count}, InstanceNorm layers: {in_count}')
    if my_parameters['compile']:
        model = torch.compile(model).to(device)
    else:
        model = model.to(device)

    # Create teacher model
    teacher_model = dl_config.setup_model(my_parameters['model'], my_parameters['encoder'])
    if my_parameters['normalization'] == 'remove':
        remove_bn_layers(teacher_model)
    elif my_parameters['normalization'] == 'in':
        replace_bn_with_in(teacher_model, exclude_modules=['encoder'])
        bn_count, in_count = count_norm_layers(model)
        print(f'BatchNorm layers: {bn_count}, InstanceNorm layers: {in_count}')
    if my_parameters['compile']:
        teacher_model = torch.compile(teacher_model)
        teacher_model.load_state_dict(model.state_dict())

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
        project=my_parameters['project_name'],
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
        wandb.define_metric('conf_ratio', summary='mean')
    else:
        wandb.define_metric('epoch', summary='max')
        wandb.define_metric('total_loss', summary='min')
        wandb.define_metric('val_loss', summary='min')

    return TrainingContext(
        model=model,
        teacher_model=teacher_model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        transform_train=transform_train,
        transform_val=transform_val,
        logger=mylogger,
        my_parameters=my_parameters
    )

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

def prepare_data(my_parameters, transform_train, transform_val):
    labeled_data, labels, unlabeled_data, padding_info, unlabeled_padding_info = dl_config.load_and_preprocess_data()

    # Apply window adjustment to all data images (0.45-0.55 -> 0-1), only for data_image
    print("Applying window adjustment to images (0.45-0.55 -> 0-1)...")

    # Process labeled data
    adjusted_labeled_data = []
    for img in labeled_data:
        # Apply window adjustment with min=0.45, max=0.55
        adjusted_img = windows_adjustment_one_image(img, min=-0.05, max=0.05)
        adjusted_labeled_data.append(adjusted_img)

    # Process unlabeled data
    adjusted_unlabeled_data = []
    for img in unlabeled_data:
        # Apply window adjustment with min=0.45, max=0.55
        adjusted_img = windows_adjustment_one_image(img, min=-0.05, max=0.05)
        adjusted_unlabeled_data.append(adjusted_img)

    train_data, val_data, train_labels, val_labels, train_padding_info, val_padding_info = train_test_split(
        adjusted_labeled_data,  # Use adjusted data
        labels,
        padding_info,
        test_size=my_parameters['ratio'],
        random_state=my_parameters['seed'],
        shuffle=False
    )

    if my_parameters['mode'] == 'semi':
        train_data.extend(adjusted_unlabeled_data)  # Use adjusted unlabeled data
        train_labels.extend([None]*len(adjusted_unlabeled_data))
        train_padding_info = pd.concat([train_padding_info, unlabeled_padding_info], ignore_index=True)

    train_dataset = load_data.my_Dataset(train_data, train_labels, train_padding_info, transform=transform_train)
    val_dataset = load_data.my_Dataset(val_data, val_labels, val_padding_info, transform=transform_val)
    # Disable augmentation during validation
    val_dataset.set_use_transform(False)

    if my_parameters['mode'] == 'semi':
        batch_size = my_parameters['label_batch_size'] + my_parameters['unlabel_batch_size']
        labeled_ratio = my_parameters['label_batch_size'] / batch_size
        sampler = load_data.MixedRatioSampler(train_dataset, labeled_ratio, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    else:
        batch_size = my_parameters['label_batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    print(f'len of train_data: {len(train_data)}, len of val_data: {len(val_data)}')

    return train_dataset, val_dataset, train_loader, val_loader

# ------------------- Consistency Loss -------------------


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


# ------------------- Epoch -------------------

def train_one_epoch(context, epoch):
    """Trains the model for one epoch using the provided context"""
    model = context.model
    teacher_model = context.teacher_model
    device = context.device
    train_loader = context.train_loader
    my_parameters = context.my_parameters
    criterion = context.criterion
    optimizer = context.optimizer
    scaler = context.scaler

    model.train()

    # Initialize loss variables
    accumulation_steps = my_parameters['accumulation_steps']
    conf_threshold = 0.95
    supervised_total = 0.0
    confs_one_total = 0.0
    mask_one_total = 0.0
    if my_parameters['mode'] == 'semi':
        total_cons_loss = 0.0
        total_loss_total = 0.0
        alpha = 0

        rampup = my_parameters['consistency_rampup']
        rampup_weight = exp(-5 * (1 - epoch / rampup) ** 2)
        if epoch > rampup:
            rampup_weight = 1
        cons_combine_weight = my_parameters['consistency_weight'] * rampup_weight
        if epoch == 0:
            train_loader.dataset.set_teacher_model(teacher_model)

    for i, (images, labels, masks, is_unlabels) in enumerate(tqdm(train_loader)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).bool()
        is_unlabels = is_unlabels.to(device, non_blocking=True)

        original_masks = masks.float()

        with autocast(device_type='cuda'):
            outputs = model(images)
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)

        if my_parameters['mode'] == 'supervised':
            supervised_loss = criterion(outputs, labels, masks)
            supervised_loss = supervised_loss / accumulation_steps
            scaler.scale(supervised_loss).backward()
            supervised_total += supervised_loss.item()
        elif my_parameters['mode'] == 'semi':
            one_indices = torch.nonzero(is_unlabels).squeeze(1)
            zero_indices = torch.nonzero(~is_unlabels).squeeze(1)

            supervised_loss = criterion(outputs[zero_indices], labels[zero_indices], masks[zero_indices])
            supervised_loss = supervised_loss / accumulation_steps

            # Conf should be where labels > conf_threshold or labels < (1 - conf_threshold)
            confs = torch.where((labels > conf_threshold) | (labels < (1 - conf_threshold)), 1, 0).float()
            confs_one_total += (confs * original_masks).sum().item()
            mask_one_total += original_masks.sum().item()
            masks = confs * masks

            cons_loss = criterion(outputs[one_indices], labels[one_indices], masks[one_indices])
            cons_loss = cons_loss / accumulation_steps

            total_loss = supervised_loss * (1 - cons_combine_weight) + cons_loss * cons_combine_weight

            scaler.scale(total_loss).backward()
            
            supervised_total += supervised_loss.item()
            total_cons_loss += cons_loss.item()
            total_loss_total += total_loss.item()

        if (i+1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update teacher model using EMA
        if my_parameters['mode'] == 'semi':
            if epoch < my_parameters['teacher_alpha_initial_epoch']:
                teacher_model.load_state_dict(model.state_dict())
            elif epoch <= my_parameters['teacher_alpha_mid_epoch']:
                alpha = my_parameters['teacher_alpha_mid']
                update_ema_variables(teacher_model, model, alpha=alpha)
            else:
                alpha = my_parameters['teacher_alpha']
                update_ema_variables(teacher_model, model, alpha=alpha)
            train_loader.dataset.set_teacher_model(teacher_model)

    # If the number of batches is not a multiple of accumulation_steps, step the optimizer
    if len(train_loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # For each epoch, divide the total loss by the number of samples
    train_loss_m = supervised_total / len(train_loader) * accumulation_steps
    if my_parameters['mode'] == 'semi':
        total_cons_loss_m = total_cons_loss / len(train_loader) * accumulation_steps
        total_loss_m = total_loss_total / len(train_loader) * accumulation_steps
        conf_ratio_m = (confs_one_total / mask_one_total) if mask_one_total > 0 else 0.0
    else:
        total_loss_m = train_loss_m
        conf_ratio_m = None

    if my_parameters['mode'] == 'semi':
        return train_loss_m, total_cons_loss_m, total_loss_m, alpha, conf_ratio_m
    else:
        return None, None, total_loss_m, None, conf_ratio_m

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0

    # Update validation loop autocast
    with torch.no_grad(), autocast(device_type='cuda'):
        for images, labels, masks, _ in tqdm(val_loader):
            # Add non_blocking=True to allow overlapping data transfer and compute
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).bool()

            outputs = model(images)
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels, masks)

            val_loss += loss.item()

    val_loss_mean = val_loss / len(val_loader)
    return val_loss_mean

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
    context = setup_environment(my_parameters)
    register_signals()

    # Extract needed values from context
    model = context.model
    teacher_model = context.teacher_model
    device = context.device
    optimizer = context.optimizer
    scheduler = context.scheduler
    criterion = context.criterion
    scaler = context.scaler
    transform_train = context.transform_train
    transform_val = context.transform_val
    mylogger = context.logger

    train_dataset, val_dataset, train_loader, val_loader = prepare_data(my_parameters, transform_train, transform_val)

    train_loss_best = float('inf')
    val_loss_best = float('inf')
    val_teacher_loss_best = float('inf')
    no_improvement_count = 0
    soft_dice_list: List[float] = []

    try:
        for epoch in range(my_parameters['n_epochs']):

            print(f"Epoch {epoch} of {my_parameters['n_epochs']}")

            # ------------------- Training -------------------

            # Create training context
            context = TrainingContext(
                model=model,
                teacher_model=teacher_model,
                device=device,
                train_loader=train_loader,
                my_parameters=my_parameters,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler
            )

            supervised_loss_m, cons_loss_m, total_loss_m, alpha, conf_ratio_m = train_one_epoch(context, epoch)

            # ------------------- Validation -------------------

            val_loss_mean = validate(model, device, val_loader, criterion)
            if my_parameters['mode'] == 'semi':
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
                    'supervised_loss': supervised_loss_m,
                    'cons_loss_un': cons_loss_m,
                    'total_loss': total_loss_m,
                    'val_loss': val_loss_mean,
                    'val_teacher_loss': val_teacher_loss_mean,
                    'learning_rate': current_lr,
                    'alpha': alpha,
                    'conf_ratio': conf_ratio_m
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
            if total_loss_m < train_loss_best:
                train_loss_best = total_loss_m
                print(f'New best training loss: {train_loss_best:.3f}')
            
            if my_parameters['mode'] == 'semi':
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
