# EEG-Based MNIST Digit Classification

This repository implements an EEG-based digit classification pipeline using the VisualMNIST dataset. The goal is to classify which digit (0–9) a subject is viewing from EEG signals, plus an additional “black screen” class (no digit stimulus).

The main work is provided as a single end-to-end Jupyter notebook that covers data preparation, model training, evaluation, interpretability, and a final baseline comparison section to demonstrate that the proposed model performs better than standard EEG baselines.

## Dataset Used

VisualMNIST (EEG)

https://mindbigdata.com/opendb/visualmnist.html

## Repository Contents

- `eeg_mnist_classification.ipynb`: Main notebook containing the full workflow.
- `DualAttentionNet.py`: Reference implementation pieces related to the proposed dual-attention EEGNet-style architecture (used as a model component and for experimentation).
- `LICENSE`: Project license.

## Project Overview

### Problem
Given an EEG window recorded while a subject is shown a digit stimulus, predict the digit class.

### Approach
The notebook implements:

- Data loading and preprocessing
  - Parsing the dataset
  - Train/validation/test splitting
  - Tensor formatting suitable for EEG deep learning models

- A proposed model
  - EEGNet-style backbone
  - Dual attention blocks (channel-focused attention)
  - Regularization and training improvements such as class weighting and modern optimizers
  - Test-time augmentation (TTA) at evaluation for improved robustness

- Baseline comparison
  - Trains multiple established EEG baseline architectures where available
  - Evaluates and compares the proposed approach against those baselines

- Evaluation and reporting
  - Confusion matrix
  - Per-class precision/recall/F1
  - Additional comparison metrics (e.g., Cohen’s kappa, ROC-AUC where applicable, top-k accuracy, inference time)

- Interpretability
  - Saliency-based visualization to understand which time regions influence predictions

## How to Run

### Option A: Run in Jupyter
1. Create and activate a Python environment.
2. Install dependencies.
3. Open and run the notebook top-to-bottom.

### Recommended environment
- Python 3.9+ (3.10 or 3.11 usually works well)
- macOS / Linux / Windows supported
- Optional: CUDA-capable GPU for faster training

### Install dependencies
If you do not already have the required libraries, install them using pip.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core scientific stack
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Deep learning
pip install torch torchvision torchaudio

# EEG models (optional but recommended for baseline comparison)
pip install braindecode

# Notebook runtime
pip install jupyter ipykernel
```

Notes:
- If `braindecode` is not installed or some models are unavailable, the notebook is designed to fall back to alternative baselines.
- For GPU support, install the appropriate PyTorch build from the official PyTorch website.

### Launch the notebook
```bash
jupyter notebook
```
Then open `eeg_mnist_classification.ipynb` and run the cells in order.

## Key Outputs

The notebook produces:

- Training and validation learning curves
- Test accuracy (with and without test-time augmentation)
- Confusion matrix (counts and normalized)
- Per-class metrics table
- Model comparison tables and plots
- Additional “model comparison” metrics to support claims that the proposed model is better

## Metrics: What “Inference (ms/sample)” Means

“Inference (ms/sample)” is the average time (in milliseconds) the model takes to produce a prediction for one EEG sample/window, measured during evaluation. Lower values indicate faster prediction.

Because runtime depends on CPU/GPU, batch size, and system load, use it as a relative comparison within the same environment.

## Reproducibility Tips

- Run the notebook from a clean kernel start.
- Keep device settings consistent (CPU vs GPU) when comparing inference time.
- If you change augmentation/TTA settings, report them alongside final scores.

## Citation

If you use this work in an academic context, cite the VisualMNIST dataset source:

- MindBigData VisualMNIST: https://mindbigdata.com/opendb/visualmnist.html
