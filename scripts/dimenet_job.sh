#!/bin/bash
#BSUB -q gpua100
#BSUB -J dimenet2_MT
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o out/dimenet_MT_%J.out
#BSUB -e err/dimenet_MT_%J.err

# Set environment variables for optimization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your training (removed compile_model=true)
python src/run.py model=dimenetpp trainer=mean-teacher \
  dataset.init.batch_size_train=384 \
  dataset.init.num_workers=8 \
  trainer.train.total_epochs=200 \
  trainer.train.validation_interval=20 \
  trainer.init.optimizer.lr=0.0015