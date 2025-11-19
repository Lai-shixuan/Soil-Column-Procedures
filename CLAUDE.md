# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based soil column image analysis toolchain with deep learning components. The system processes soil column images through a structured pipeline to perform segmentation, thresholding, and analysis. The codebase is organized into API functions, workflow tools, tests, and documentation.

## Repository Structure

- **src/API_functions/**: Core reusable API modules
  - **DL/**: Deep learning utilities (load_data.py, evaluate.py, multi_input_adapter.py, shape_base.py, etc.)
  - **Images/**: Image I/O and batch processing (file_batch.py, file_single.py, read_raw.py)
  - **Soils/**: Soil-specific processing (column_analysis.py, pre_process.py, threshold_*.py)
  - **Visual/**: Visualization and comparison tools
  - **DICOM/**: DICOM file support (limited use, superseded by microDicom viewer)

- **src/workflow_tools/**: End-to-end workflow scripts and utilities
  - **database/**: Data processing pipeline scripts (s1-7*.py)
  - **model_online/**: Online model serving (fr_unet.py, mcc.py)
  - **cvat_noisy/**: CVAT integration for noisy labels
  - **UI/**: Image viewing and scrolling interfaces
  - **one-click-process-script.py**: Unified processing script

- **tests/**: Unit tests organized by domain (DL/, Soils/)

## High-Level Architecture

The project follows a modular architecture with three core layers:

1. **API Layer (src/API_functions/)**:
   - Atomic, reusable components organized by domain
   - **DL/**: Deep learning utilities (data loading, evaluation, model adapters)
   - **Images/**: Image I/O and batch processing
   - **Soils/**: Soil-specific processing (column analysis, preprocessing, thresholding)
   - **Visual/**: Visualization tools
   - **DICOM/**: Legacy DICOM support

2. **Workflow Layer (src/workflow_tools/)**:
   - Combines API components into end-to-end workflows
   - **database/**: 7-step data processing pipeline (s1-s7 scripts)
   - **model_online/**: Online model serving
   - **cvat_noisy/**: CVAT integration for annotation
   - **UI/**: User interfaces for image viewing

3. **Execution Layer**:
   - Tests (tests/): Unit tests organized by domain
   - Documentation (doc/): Demo scripts and documentation
   - One-click processing script: Unified entry point for pipelines

## Data Processing Pipeline

The system uses a structured 6-step pipeline for each soil column part:

1. **0.Origin** - Raw input images
2. **1.Reconstruct** - Reconstructed/processed images
3. **2.ROI** - ROI-cropped images
4. **3.Rename** - Renamed and indexed images
5. **4.Threshold** - Thresholded/segmented images
6. **5.Analysis** - Final analysis results

Each soil column is divided into multiple parts, with a standardized folder structure:
```
{column_id}-{part_id:02d}/
├── 0.Origin/
├── 1.Reconstruct/
├── 2.ROI/
├── 3.Rename/
├── 4.Threshold/
└── 5.Analysis/
```

## Common Development Tasks

### Running Tests
```bash
# Run all tests
python -m pytest

# Run tests with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/DL/test_multi_input_adapter.py

# Run specific test
python -m pytest tests/DL/test_multi_input_adapter.py::test_function_name
```

### Linting and Code Quality
Currently, there are no explicit linting configurations (flake8, pylint, etc.) checked into the repository. However, it's recommended to follow Python best practices.

### Installing Dependencies
The project uses Python with PyTorch as the deep learning framework. Dependencies are typically managed via pip. Key requirements include:
- pytorch
- opencv-python
- numpy
- pandas
- tqdm
- wandb

For specific environments, check wandb/run-*/files/requirements.txt for generated dependency lists.

### Processing Workflows

**One-click processing script** (src/workflow_tools/one-click-process-script.py):
- Orchestrates the complete image processing pipeline
- Manages column parts, ROI selection, and batch operations
- Use: Initialize, check, or process soil columns

**Database processing scripts** (src/workflow_tools/database/s*.py):
- s1*: Raw data cleaning and PNG conversion
- s2*: ROI (Region of Interest) cutting and shape processing
- s3*: Data harmonization across samples
- s4*: Data splitting, augmentation, refinement, conversion, and padding
- s5*: Preprocessing and thresholding operations
- s6*: Data mixup and precheck validation
- s7*: Analysis (average gray value, longitudinal section generation)

### Key Classes and Components

**SoilColumn** (src/API_functions/Soils/column_batch_process.py):
- Manages soil column data structure
- Handles part organization and ROI loading
- Creates standardized folder structures
- ROI data stored in CSV files: `roi-{column_id}-{part_id}.csv`

**ImageName** (src/API_functions/Images/file_batch.py):
- Manages image naming conventions with prefix/suffix
- Standardizes file naming across pipeline

**roi_region** (src/API_functions/Images/file_batch.py):
- Defines region of interest (x1, y1, width, height, z1, z2)
- Used for cropping and processing

**my_Dataset** (src/API_functions/DL/load_data.py):
- PyTorch Dataset for soil column images
- Supports labeled/unlabeled data
- Handles padding information and transformations
- Implements image augmentation via ImageAugmenter

## Deep Learning Components

**Training and Inference** (src/workflow_tools/DL_Inference.py, DL-Net.py):
- Model inference pipelines
- Online model serving capabilities

**Multi-input Adapter** (src/API_functions/DL/multi_input_adapter.py):
- Handles multiple input modalities
- Used in test suite (tests/DL/test_multi_input_adapter.py)

**Data Augmentation** (src/workflow_tools/database/s4augmented_labels.py):
- ImageAugmenter class for training data augmentation
- Integrated with PyTorch transforms

## Key Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV (cv2)**: Image processing
- **NumPy**: Numerical operations
- **Pandas**: DataFrame operations
- **tqdm**: Progress bars
- **Weights & Biases (wandb)**: Experiment tracking

## Important Notes

- The project uses absolute paths in some modules (e.g., file_batch.py:11) - be aware when running in different environments
- ROI information is stored in CSV files alongside image data
- The system was designed for Chinese documentation (README.md) but code uses English naming
- Some code paths are Windows-specific (e.g., hardcoded C: drive paths in file_batch.py)
- CVAT noisy label tools are available but may need configuration
- One-click processing script is provided but may be outdated with current database structure

## Testing

- Uses pytest with CLI logging enabled (see pytest.ini)
- Tests are organized by domain in tests/ directory
- DL tests focus on multi-input adapters and ROI processing
- Soils tests cover preprocessing and thresholding methods

## Development Patterns

- API_functions are designed as atomic, reusable components
- workflow_tools combine APIs into end-to-end workflows
- Standardized 6-stage pipeline across all processing
- CSV-based configuration for ROI and metadata
- PyTorch-based deep learning with augmentation support
