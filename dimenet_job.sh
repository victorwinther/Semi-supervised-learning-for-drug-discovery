#!/bin/bash
#BSUB -q c02516            # Queue name (choose gpua100, gpuv100, gpul40s, etc.)
#BSUB -J dimenet2      # Job name
#BSUB -n 4                  # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process"  # One GPU exclusively
#BSUB -W 12:00              # Walltime limit (HH:MM, max 24 hours for GPU queues)
#BSUB -R "rusage[mem=8GB]"  # System RAM per CPU core
#BSUB -o out/dimenet_%J.out     # Standard output file
#BSUB -e err/dimenet_%J.err     # Standard error file


# Navigate to your project directory
#cd ~/Semi-supervised-learning-for-drug-discovery

# Run the training script
HYDRA_FULL_ERROR=1 python src/run.py   model=dimenetpp   dataset.init.batch_size_inference=128   dataset.init.batch_size_train=64
