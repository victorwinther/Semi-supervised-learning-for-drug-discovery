#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua40
### -- set the job Name --
#BSUB -J dimenet_best_final
### -- ask for number of cores (Match this to num_workers!) --
#BSUB -n 8
### -- force all 4 cores to be on the same node (Crucial for speed) --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
#BSUB -W 1:00
### -- request 16GB of system-memory (Safe headroom for full data) --
#BSUB -R "rusage[mem=32GB]"
### -- output and error logs --
#BSUB -o out/hero_%J.out
#BSUB -e err/hero_%J.err


echo "Job running on node:"
hostname

echo "GPU Status:"
nvidia-smi

module load cuda/11.8

export OMP_NUM_THREADS=8

echo "Starting Training..."


source ../DeepLearning/venv/bin/activate

python src/run.py \
  model=dimenetpp \
  dataset.init.batch_size_train=256 \
  dataset.init.batch_size_inference=2048 \
  dataset.init.num_workers=8 \
  logger.group="full_dataset_hero" \
  logger.name="best_test_run" \
  trainer.train.total_epochs=50 \
  model.init.hidden_channels=128

echo "Done!"