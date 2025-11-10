#!/bin/bash
#BSUB -q gpua100            # Queue name (choose gpua100, gpuv100, gpul40s, etc.)
#BSUB -J dimenet_train      # Job name
#BSUB -n 4                  # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process"  # One GPU exclusively
#BSUB -W 12:00              # Walltime limit (HH:MM, max 24 hours for GPU queues)
#BSUB -R "rusage[mem=8GB]"  # System RAM per CPU core
#BSUB -R "select[gpu80gb]"  # Request 80 GB A100 (omit or adjust for other queues)
#BSUB -o dimenet_%J.out     # Standard output file
#BSUB -e dimenet_%J.err     # Standard error file


# Navigate to your project directory
#cd ~/Semi-supervised-learning-for-drug-discovery

# Run the training script
HYDRA_FULL_ERROR=1 python src/run.py model=dimenet logger.disable=true
