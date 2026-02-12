# IMTalker Custom Training Guide & Changelog

This document summarizes the changes, fixes, and the operational pipeline established for training IMTalker on a custom dataset (15k+ videos) using 4x NVIDIA A100 GPUs.

## 🚀 Training Pipeline Overview

The complete pipeline is orchestrated by `launch_full_training.sh` and consists of four main phases:

1.  **Data Preprocessing (Phase 1 & 2)**:
    - Extracts facial landmarks (`lmd`) and frames from raw MP4s.
    - Pre-extracts features (Audio, Motion, Gaze, Smirk/Pose) for the Generator.
    - Optimized to run on all 4 GPUs with 32 concurrent workers (8 per GPU).

2.  **Renderer Training (Phase 3)**:
    - Trains the image-to-image renderer.
    - Configured for 4-GPU DDP (Distributed Data Parallel).
    - **Batch Size**: 4 per GPU.
    - **Monitoring**: Saves `loss_curve.png` in the experiment folder.

3.  **Generator Training (Phase 4)**:
    - Starts automatically after the Renderer finishes.
    - **Batch Size**: 16 per GPU.
    - **Monitoring**: Saves `loss_curve.png` in the experiment folder.

## 🛠 Technical Changes & Fixes

### 1. Core Code Stability
- **Dimension Mismatch (`generator/FM.py`)**: Removed an unnecessary `.squeeze()` that was causing concatenation errors in the Transformer blocks.
- **Key Mismatch (`generator/dataset.py`)**: Aligned dataset keys (`gaze`, `pose`, `cam`) with the model's expected inputs.
- **Validation Fix (`renderer/train.py`)**: Correctly unpacked the tuple returned by the generator during the validation step.
- **Typo Fix (`generator/train.py`)**: Corrected `--dataset_pat` argument to `--dataset_path`.

### 2. Dataset Logic
- **Dynamic Splitting (`renderer/dataset.py`)**: Replaced hardcoded video counts with a relative 95% train / 5% validation split to handle datasets of any size.
- **Preprocessing Limits**: Removed hardcoded "600 video" limits in `prepare_renderer_data.py` and `prepare_generator_data.py` to allow full dataset (15k+) training.

### 3. Monitoring & UX
- **Loss Plotting**: Added a `LossPlotterCallback` to both training scripts that generates PNG graphs of training progress every 1,000 steps.
- **Logging**: Improved error reporting in preprocessing scripts to help identify corrupt video files.

### 4. Infrastructure & Environment
- **Dependency Resoltuion**: Installed missing `opencv-python`, `einops`, and `matplotlib`.
- **Conda Lock**: All scripts are configured to automatically use the `IMTalker` conda environment.

## 📈 Suggested Commands

### Full Pipeline Restart
```bash
# Ensure you are at the project root
bash /workspace/launch_full_training.sh &
```

### Monitoring Progress
- **Pre-processing Logs**: `tail -f /workspace/full_prep_renderer.log`
- **Training Logs**: `tail -f /workspace/renderer_train.log`
- **TensorBoard**: `tensorboard --logdir ./exps`

### Viewing Loss Curves
Check the following paths for periodic graph updates:
- `/workspace/exps/custom_renderer/loss_curve.png`
- `/workspace/exps/custom_generator/loss_curve.png`
