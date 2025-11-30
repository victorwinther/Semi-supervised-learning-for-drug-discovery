#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpul40s
### -- set the job Name --
#BSUB -J dimenet_hero
### -- ask for number of cores (Match this to num_workers!) --
#BSUB -n 8
### -- force all 4 cores to be on the same node (Crucial for speed) --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- request 16GB of system-memory (Safe headroom for full data) --
#BSUB -R "rusage[mem=32GB]"
### -- output and error logs --
#BSUB -o out/hero_%J.out
#BSUB -e err/hero_%J.err

# -- end of LSF options --

echo "Job running on node:"
hostname

echo "GPU Status:"
nvidia-smi

# Load CUDA (Matches the guide's recommendation for A100)
module load cuda/11.8

# Set environment variables for speed
export OMP_NUM_THREADS=8

echo "Starting Training..."

# The "Hero" Command
# Optimized for Speed (workers=4, inf_batch=2048)
# Optimized for Accuracy (noise=0.01, unsup=5.0, decay=0.999)

python src/run.py \
    model=dimenetpp trainer=mean-teacher \
  dataset.init.batch_size_train=256 \
  dataset.init.batch_size_inference=2048 \
  dataset.init.num_workers=8 \
  trainer.init.augment_coords=true \
  trainer.init.coord_noise_std=0.05 \
  trainer.init.unsup_weight=1.0 \
  trainer.init.ema_decay=0.999 \
  trainer.init.optimizer.weight_decay=0.005 \
  logger.group="full_dataset_hero" \
  logger.name="best_test_run" \
  trainer.train.total_epochs=250 \
  model.init.hidden_channels=64 


echo "Done!"