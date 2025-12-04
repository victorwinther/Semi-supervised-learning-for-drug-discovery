# Semi-Supervised Learning for Drug Discovery

Semi-supervised learning methods for molecular property prediction on the QM9 dataset. Developed for the Deep Learning course at DTU.

## Overview

This project explores semi-supervised learning techniques to predict HOMO energy from molecular graphs, using only 8% labeled data while leveraging 72% unlabeled data.

**Methods implemented:**

- Mean Teacher with EMA
- n-CPS (Cross Pseudo Supervision)
- Consistency Regularization with augmentations

**Models:**

- GCN, GIN (2D graph-based)
- SchNet, DimeNet++ (3D coordinate-based)

## Installation

```bash
# Install PyTorch first (see https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# Install PyTorch Geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
pip install torch_geometric

# Install remaining dependencies
pip install -r requirements.txt
```

## Usage

Training uses Hydra for configuration:

```bash
# Run with default config
python src/run.py

# Specify model and trainer
python src/run.py model=dimenetpp trainer=mean-teacher

# Override hyperparameters
python src/run.py trainer.train.total_epochs=100 trainer.init.ema_decay=0.999
```

Available models: `gcn`, `gin`, `schnet`, `dimenetpp`

Available trainers: `mean-teacher`, `n-cps`, `consistency-aug`, `semi-supervised-ensemble`

## Reproducing Results

Run the notebook `reproduce_results.ipynb` to reproduce our best results (Val MSE: 0.0124, Test MSE: 0.0133) using the pretrained model checkpoint.

## Project Structure

```
├── configs/          # Hydra configuration files
│   ├── model/        # Model architectures
│   ├── trainer/      # Training methods
│   └── dataset/      # Dataset settings
├── src/
│   ├── run.py        # Main entry point
│   ├── models.py     # GNN model definitions
│   ├── trainer.py    # Semi-supervised trainers
│   └── qm9.py        # Data loading
└── reproduce_results.ipynb
```
