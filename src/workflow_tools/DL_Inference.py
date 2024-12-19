# Import necessary libraries
import numpy as np
import torch
import sys
import albumentations as A
import segmentation_models_pytorch as smp
import pandas as pd
import logging
import cv2
import hashlib

# sys.path.insert(0, "/root/Soil-Column-Procedures")
sys.path.insert(0, "c:/Users/laish/1_Codes/Image_processing_toolchain/")

from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from src.API_functions.DL import load_data, evaluate
from src.API_functions.Images import file_batch as fb
from src.API_functions.Soils import pre_process


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class InferenceConfig:
    """Configuration dataclass for inference pipeline.
    
    Attributes:
        model_type: Type of model to use (DeepLabV3Plus, Unet, or PSPNet).
        model_path: Path to the model weights file.
        backbone: Backbone to use for the model.
        device: Device to run inference on ('cuda' or 'cpu').
        mode: Mode of operation ('evaluation' or 'inference').
        images_path: Directory containing input images.
        labels_path: Directory containing label images (optional, required for evaluation mode).
        save_path: Directory to save outputs and metrics.
        batch_size: Number of images to process in each batch.
        run_config: Dictionary containing configuration parameters:
            summary_filename: Name of the summary metrics file.
        remove_prefix: Whether to remove '_orig_mod.' prefix from state_dict keys.
    """
    model_type: str
    model_path: str
    backbone: str  # New parameter for model backbone
    device: str
    mode: str
    images_path: str
    labels_path: Optional[str]
    save_path: str
    batch_size: int
    run_config: Dict[str, str]  # Only needs 'summary_filename'
    remove_prefix: bool = False  # Add this new parameter with default False

    def validate(self) -> None:
        """Validates the configuration parameters.

        Raises:
            ValueError: If model_type is invalid or labels_path is missing in evaluation mode.
            FileNotFoundError: If any required paths don't exist.
            RuntimeError: If CSV files are not accessible.
        """
        valid_model_types = {'DeepLabV3Plus', 'Unet', 'PSPNet'}
        if self.model_type not in valid_model_types:
            raise ValueError(f"Model type must be one of {valid_model_types}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        if not Path(self.images_path).exists():
            raise FileNotFoundError(f"Images path does not exist: {self.images_path}")
        
        if not Path(self.save_path).exists():
            raise FileNotFoundError(f"Save path does not exist: {self.save_path}")
        
        if self.mode == 'evaluation':
            if not self.labels_path or not Path(self.labels_path).exists():
                raise ValueError("Labels path must be provided and exist in evaluation mode")
            
            # Check CSV files accessibility
            csv_files = [
                Path(self.save_path) / "detailed_metrics.csv",
                Path(self.save_path) / self.run_config['summary_filename']
            ]
            
            for csv_file in csv_files:
                if csv_file.exists():
                    try:
                        with open(csv_file, 'a') as f:
                            pass
                    except PermissionError:
                        raise RuntimeError(f"Cannot access {csv_file}. Please close any applications that might be using this file.")
        
        valid_backbones = {'efficientnet-b0', 'resnet34', 'resnet50', 'mobilenet_v2'}
        if self.backbone not in valid_backbones:
            raise ValueError(f"Backbone must be one of {valid_backbones}")


class InferencePipeline:
    """Main class for handling deep learning inference.
    
    This class manages the entire inference pipeline, including model setup,
    image processing, metrics calculation, and results saving.

    Attributes:
        config: Configuration object containing all pipeline parameters.
        model: The loaded deep learning model.
        transform: Image transformation pipeline.
        logger: Logging object for pipeline messages.
        run_timestamp: Timestamp string for the current run.
        run_id: Unique identifier for the current run.
        output_dir: Directory for saving current run outputs.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.config.validate()
        
        self.model = self._setup_model()
        self.transform = self._get_transform()
        self.logger = logging.getLogger(__name__)
        self.run_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = self._generate_run_id()
        self.output_dir = self._setup_output_directory()

    def _generate_run_id(self) -> str:
        """Generates a unique run ID combining timestamp and hash.
        
        Returns:
            str: Unique run identifier in format 'run_YYYYMMDD_HHMMSS_hash'.
        """
        # Create a hash from timestamp and a random number for uniqueness
        hash_input = f"{self.run_timestamp}_{np.random.randint(0, 1000000)}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"run_{self.run_timestamp}_{hash_suffix}"

    def _setup_output_directory(self) -> Path:
        """Creates and returns the run-specific output directory.
        
        Returns:
            Path: Path object pointing to the created output directory.
        """
        output_dir = Path(self.config.save_path) / f"run_{self.run_timestamp}"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def _setup_model(self) -> torch.nn.Module:
        """Sets up and returns the model for inference.
        
        Returns:
            torch.nn.Module: Initialized and loaded model ready for inference.
        """
        model_mapping = {
            'DeepLabV3Plus': lambda: smp.DeepLabV3Plus(
                encoder_name=self.config.backbone,
                encoder_weights="imagenet",
                in_channels=1,
                classes=1
            ),
            'Unet': lambda: smp.Unet(
                encoder_name=self.config.backbone,
                encoder_weights="imagenet",
                in_channels=1,
                classes=1
            ),
            'PSPNet': lambda: smp.PSPNet(
                encoder_name=self.config.backbone,
                encoder_weights="imagenet",
                in_channels=1,
                classes=1
            )
        }
        
        model = model_mapping[self.config.model_type]()

        # Load the state dictionary and fix the keys
        state_dict = torch.load(self.config.model_path, map_location=self.config.device)
        
        if self.config.remove_prefix:
            # Remove '_orig_mod.' prefix from keys if requested
            fixed_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                fixed_state_dict[new_key] = value
            state_dict = fixed_state_dict

        model.load_state_dict(state_dict)
        return model.to(self.config.device).eval()

    @staticmethod
    def _get_transform() -> A.Compose:
        return A.Compose([
            ToTensorV2()
        ])

    def _extract_model_log(self) -> str:
        """Extracts the model log information from the model path filename.
        
        Returns:
            str: The part of the model filename after the second underscore.
        """
        model_filename = Path(self.config.model_path).name
        parts = model_filename.split('_', 2)
        return parts[2].replace('.pth', '') if len(parts) > 2 else ''

    def process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                    image_paths: List[str], 
                    batch_start_idx: int) -> List[Dict]:
        """Processes a batch of images through the model.
        
        Args:
            batch: Tuple of (images, labels) tensors.
            image_paths: List of paths to the original images.
            batch_start_idx: Starting index of the current batch.

        Returns:
            List[Dict]: List of metrics dictionaries for each processed image.
        """
        images, labels = [x.to(self.config.device) for x in batch]
        output = self.model(images)
        
        if output.dim() == 4 and output.size(1) == 1:
            output = output.squeeze(1)
        
        output_prob = torch.sigmoid(output)
        
        metrics = []
        for j in range(images.size(0)):
            batch_idx = batch_start_idx + j
            if batch_idx >= len(image_paths):
                break
                
            metrics.append(self._process_single_image(
                output_prob[j], output[j], labels[j], 
                image_paths[batch_idx]
            ))
            
        return metrics

    def _process_single_image(self, 
                            output_prob: torch.Tensor, 
                            output: torch.Tensor, 
                            label: torch.Tensor, 
                            image_path: str) -> Dict:
        """Processes a single image and calculates its metrics.
        
        Args:
            output_prob: Probability prediction tensor.
            output: Raw model output tensor.
            label: Ground truth label tensor.
            image_path: Path to the original image.

        Returns:
            Dict: Dictionary containing all calculated metrics for the image.
        """
        image_name = Path(image_path).name
        
        # Convert to binary only for saving
        output_save = (output_prob.cpu().numpy() > 0.5).astype(np.uint8) * 255
        save_path = self.output_dir / image_name
        cv2.imwrite(str(save_path), output_save)
        
        metrics = {
            'run_id': self.run_id,
            'image_name': image_name,
            'model_type': self.config.model_type,
            'model_log': self._extract_model_log(),
            'timestamp': pd.Timestamp.now()
        }

        tp, fp, fn, tn = evaluate.get_confusion_matrix_elements(output_prob, label)
        
        # Add evaluation metrics if in evaluation mode
        if self.config.mode == 'evaluation':
            metrics.update({
                'dice_score': evaluate.dice_coefficient(output_prob, label),
                'iou_score': evaluate.iou(output_prob, label),
                'bce_loss': torch.nn.BCEWithLogitsLoss()(output, label).item(), # Becase output is not sigmoid
                'f1_score': smp.metrics.functional.f1_score(tp, fp, fn, tn, reduction='micro').item(),
                'precision': smp.metrics.functional.precision(tp, fp, fn, tn, reduction='micro').item(),
                'recall': smp.metrics.functional.recall(tp, fp, fn, tn, reduction='micro').item(),
                'accuracy': smp.metrics.functional.accuracy(tp, fp, fn, tn, reduction='micro').item()
            })
        
        return metrics

    def _save_metrics(self, metrics_df: pd.DataFrame) -> None:
        """Saves evaluation metrics to CSV files.
        
        Saves or updates two files:
        1. detailed_metrics.csv: Contains metrics for each individual image
        2. inference_summary.csv: Contains aggregated metrics for each run

        Args:
            metrics_df: DataFrame containing metrics for all processed images.
        """
        if self.config.mode != 'evaluation':
            return

        save_path = Path(self.config.save_path)
        detailed_metrics_file = save_path / "detailed_metrics.csv"
        summary_file = save_path / self.config.run_config['summary_filename']

        # Ensure model_log is in the fourth column position for both DataFrames
        column_order = ['run_id', 'image_name', 'model_type', 'model_log', 'timestamp']
        if self.config.mode == 'evaluation':
            column_order.extend(['dice_score', 'iou_score', 'bce_loss', 'f1_score', 'precision', 'recall', 'accuracy'])
        
        metrics_df = metrics_df.reindex(columns=column_order)

        # Save/Update detailed metrics
        if detailed_metrics_file.exists():
            existing_df = pd.read_csv(detailed_metrics_file)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df = updated_df.reindex(columns=column_order)
            updated_df.to_csv(detailed_metrics_file, index=False)
        else:
            metrics_df.to_csv(detailed_metrics_file, index=False)
        
        # Update summary file
        summary = {
            'run_id': self.run_id,
            'timestamp': pd.Timestamp.now(),
            'model_type': self.config.model_type,
            'model_log': self._extract_model_log(),
            'num_images': len(metrics_df),
            'avg_dice_score': metrics_df['dice_score'].mean(),
            'avg_iou_score': metrics_df['iou_score'].mean(),
            'avg_bce_loss': metrics_df['bce_loss'].mean(),
            'avg_f1_score': metrics_df['f1_score'].mean(),
            'avg_precision': metrics_df['precision'].mean(),
            'avg_recall': metrics_df['recall'].mean(),
            'avg_accuracy': metrics_df['accuracy'].mean()
        }
        
        summary_df = pd.DataFrame([summary])
        summary_column_order = ['run_id', 'timestamp', 'model_type', 'model_log', 
                              'num_images', 'avg_dice_score', 'avg_iou_score', 'avg_bce_loss', 'avg_f1_score', 'avg_precision', 'avg_recall', 'avg_accuracy']
        summary_df = summary_df.reindex(columns=summary_column_order)

        if summary_file.exists():
            existing_df = pd.read_csv(summary_file)
            updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
            updated_df = updated_df.reindex(columns=summary_column_order)
            updated_df.to_csv(summary_file, index=False)
        else:
            summary_df.to_csv(summary_file, index=False)

    def run(self) -> pd.DataFrame:
        """Executes the complete inference pipeline.
        
        This method:
        1. Loads and preprocesses images and labels
        2. Runs inference through the model
        3. Calculates metrics
        4. Saves results and metrics

        Returns:
            pd.DataFrame: DataFrame containing metrics for all processed images.
        """
        self.logger.info("Starting inference pipeline...")
        
        image_paths = fb.get_image_names(self.config.images_path, None, 'tif')
        images = fb.read_images(image_paths, 'gray', read_all=True)
        
        if self.config.mode == 'evaluation':
            label_paths = fb.get_image_names(self.config.labels_path, None, 'tif')
            labels = fb.read_images(label_paths, 'gray', read_all=True)
        else:   # In inference mode, labels are not needed
            labels = [np.zeros_like(img) for img in images]
        
        dataset = load_data.my_Dataset(images, labels, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        all_metrics = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                batch_metrics = self.process_batch(batch, image_paths, i * self.config.batch_size)
                all_metrics.extend(batch_metrics)
                # self.logger.info(f"Processed batch {i+1}/{len(dataloader)}")
        
        metrics_df = pd.DataFrame(all_metrics)
        self._save_metrics(metrics_df)
        
        self.logger.info("Inference pipeline completed successfully")
        return metrics_df


if __name__ == "__main__":

    # Have using preprocess equalization, be attenetion!!!

    config = InferenceConfig(
        model_type='Unet',
        backbone='efficientnet-b0',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mode='evaluation',  # 'inference' or 'evaluation
        
        # _extract_model_log will use this filename, don't change it
        model_path='src/workflow_tools/pths/model_U-Net_40.Unet-b0-halfLR.pth',

        images_path=r'g:\DL_Data_raw\version6-large\7.Final_dataset\test\image',
        labels_path=r'g:\DL_Data_raw\version6-large\7.Final_dataset\test\label',
        save_path=r'g:\DL_Data_raw\version6-large\_inference',

        batch_size=16,
        remove_prefix=True,  # Add this parameter
        run_config={
            'summary_filename': 'inference_summary.csv'  # Simplified config
        }
    )
    
    pipeline = InferencePipeline(config)
    metrics = pipeline.run()
