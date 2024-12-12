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
        device: Device to run inference on ('cuda' or 'cpu').
        mode: Mode of operation ('evaluation' or 'inference').
        images_path: Directory containing input images.
        labels_path: Directory containing label images (optional, required for evaluation mode).
        save_path: Directory to save outputs and metrics.
        batch_size: Number of images to process in each batch.
        run_config: Dictionary containing configuration parameters:
            summary_filename: Name of the summary metrics file.
    """
    model_type: str
    model_path: str
    device: str
    mode: str
    images_path: str
    labels_path: Optional[str]
    save_path: str
    batch_size: int
    run_config: Dict[str, str]  # Only needs 'summary_filename'

    def validate(self) -> None:
        """Validates the configuration parameters.

        Raises:
            ValueError: If model_type is invalid or labels_path is missing in evaluation mode.
            FileNotFoundError: If any required paths don't exist.
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
        
        if self.mode == 'evaluation' and (not self.labels_path or not Path(self.labels_path).exists()):
            raise ValueError("Labels path must be provided and exist in evaluation mode")


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
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=1,
                classes=1
            ),
            'Unet': lambda: smp.Unet(
                encoder_name="efficientnet-b0",
                encoder_weights="imagenet",
                in_channels=1,
                classes=1
            ),
            'PSPNet': lambda: smp.PSPNet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=1,
                classes=1
            )
        }
        
        model = model_mapping[self.config.model_type]()
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.config.device, weights_only=True))
        return model.to(self.config.device).eval()

    @staticmethod
    def _get_transform() -> A.Compose:
        return A.Compose([
            ToTensorV2()
        ])

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
            'timestamp': pd.Timestamp.now()
        }
        
        # Add evaluation metrics if in evaluation mode
        if self.config.mode == 'evaluation':
            metrics.update({
                'dice_score': evaluate.dice_coefficient(
                    output_prob.unsqueeze(0), 
                    label.unsqueeze(0)
                ),
                'iou_score': evaluate.iou(
                    output_prob.unsqueeze(0), 
                    label.unsqueeze(0)
                ),
                'bce_loss': torch.nn.BCEWithLogitsLoss()(
                    output.unsqueeze(0),    # Because output is not sigmoided
                    label.unsqueeze(0)
                ).item()
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

        # Save/Update detailed metrics
        if detailed_metrics_file.exists():
            existing_df = pd.read_csv(detailed_metrics_file)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(detailed_metrics_file, index=False)
        else:
            metrics_df.to_csv(detailed_metrics_file, index=False)
        
        # Update summary file
        summary = {
            'run_id': self.run_id,
            'timestamp': pd.Timestamp.now(),
            'model_type': self.config.model_type,
            'num_images': len(metrics_df),
            'avg_dice_score': metrics_df['dice_score'].mean(),
            'avg_iou_score': metrics_df['iou_score'].mean(),
            'avg_bce_loss': metrics_df['bce_loss'].mean()
        }
        
        summary_df = pd.DataFrame([summary])
        if summary_file.exists():
            existing_df = pd.read_csv(summary_file)
            updated_df = pd.concat([existing_df, summary_df], ignore_index=True)
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
        
        for i in range(len(images)):
            images[i] = pre_process.median(images[i], 3)
            images[i] = pre_process.histogram_equalization_float32(images[i])

        if self.config.mode == 'evaluation':
            label_paths = fb.get_image_names(self.config.labels_path, None, 'tif')
            labels = fb.read_images(label_paths, 'gray', read_all=True)
        else:
            labels = [np.zeros_like(img) for img in images]
        
        dataset = load_data.my_Dataset(images, labels, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        all_metrics = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch_metrics = self.process_batch(batch, image_paths, i * self.config.batch_size)
                all_metrics.extend(batch_metrics)
                self.logger.info(f"Processed batch {i+1}/{len(dataloader)}")
        
        metrics_df = pd.DataFrame(all_metrics)
        self._save_metrics(metrics_df)
        
        self.logger.info("Inference pipeline completed successfully")
        return metrics_df


if __name__ == "__main__":

    # Have using preprocess equalization, be attenetion!!!

    config = InferenceConfig(
        model_type='DeepLabV3Plus',
        model_path='src/workflow_tools/model_DeepLabv3+_27.More_data_median_hist.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu',

        mode='evaluation',  # 'inference' or 'evaluation

        images_path='g:/DL_Data_raw/version4-classes/5.precheck_test/image/',
        labels_path='g:/DL_Data_raw/version4-classes/5.precheck_test/label/',
        save_path='g:/DL_Data_raw/version4-classes/_inference/',

        batch_size=8,
        run_config={
            'summary_filename': 'inference_summary.csv'  # Simplified config
        }
    )
    
    pipeline = InferencePipeline(config)
    metrics = pipeline.run()
