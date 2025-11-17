#!/bin/bash
#BSUB -q c02516            # Queue name (choose gpua100, gpuv100, gpul40s, etc.)
#BSUB -J schnet      # Job name
#BSUB -n 4                  # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process"  # One GPU exclusively
#BSUB -W 12:00              # Walltime limit (HH:MM, max 24 hours for GPU queues)
#BSUB -R "rusage[mem=8GB]"  # System RAM per CPU core
#BSUB -o out/schnet_%J.out     # Standard output file
#BSUB -e err/schnet_%J.err     # Standard error file


# Navigate to your project directory
#cd ~/Semi-supervised-learning-for-drug-discovery

source ../DeepLearning/venv/bin/activate

HYDRA_FULL_ERROR=1 python src/run.py \
  model=schnet \
  trainer.train.total_epochs=100 \
  trainer.init.optimizer.lr=0.0005 \
  trainer.init.optimizer.weight_decay=0.0005 \
  model.init.hidden_channels=128 \
  model.init.num_interactions=6 \
  model.init.dropout=0.1 \
  model.init.add_mlp_head=true \
  model.init.mlp_hidden=128 \
  dataset.init.batch_size_train=32 \
  dataset.init.batch_size_inference=64
