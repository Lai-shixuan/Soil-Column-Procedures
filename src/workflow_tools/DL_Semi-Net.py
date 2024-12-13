import sys
import torch
import wandb
import signal

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

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
model = dl_config.setup_model().to(device)
optimizer, scheduler, criterion = dl_config.setup_training(
    model,
    my_parameters['learning_rate'],
    my_parameters['scheduler_factor'],
    my_parameters['scheduler_patience'],
    my_parameters['scheduler_min_lr']
)

# Initialize wandb
wandb.init(
    project="U-Net",
    name=my_parameters['wandb'],
    config=my_parameters,
)

# Add signal handler before training loop
def signal_handler(signum, frame):
    print("\nCaught interrupt signal. Cleaning up...")
    wandb.finish()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# ------------------- Data -------------------

data, labels = dl_config.load_and_preprocess_data()
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=my_parameters['ratio'], random_state=my_parameters['seed']
)

train_dataset = load_data.my_Dataset(train_data, train_labels, transform=transform_train)
val_dataset = load_data.my_Dataset(val_data, val_labels, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=my_parameters['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=my_parameters['batch_size'], shuffle=False)

print(f'len of train_data: {len(train_data)}, len of val_data: {len(val_data)}')

# ------------------- Training -------------------

val_loss_best = float('inf')
proceed_once = True

try:
    for epoch in range(my_parameters['n_epochs']):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)

            # Checking the dimension of the outputs and labels
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            # Only proceed once:
            if proceed_once:
                print(f'outputs.size(): {outputs.size()}, labels.size(): {labels.size()}')
                print(f'outputs.min: {outputs.min()}, outputs.max: {outputs.max()}')
                print(f'images.min: {images.min()}, images.max: {images.max()}')
                print(f'labels.min: {labels.min()}, labels.max: {labels.max()}')
                print(f'count of label 0: {(labels == 0).sum()}, count of label 1:{(labels == 1).sum()}')
                print('')
                proceed_once = False  # Set the flag to False after proceeding once
            
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss_mean = train_loss / len(train_loader.dataset)


        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                if outputs.dim() == 4 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                
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
