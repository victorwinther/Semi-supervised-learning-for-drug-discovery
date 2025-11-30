#!/bin/bash
#BSUB -q c02516
# 4 methods × 1 hyperparam setting = 4 jobs
#BSUB -J "qm9_dimenet_ssl[1-4]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/qm9_dimenet_ssl_%J_%I.out
#BSUB -e err/qm9_dimenet_ssl_%J_%I.err

# source ~/.venvs/ssl/bin/activate

########################
# 1) Fixed model & HPs
########################

MODEL="dimenetpp"

# Plug in the *best* LR/WD you found in Stage 1 for dimenetpp:
LR=0.001
WD=0.0005

EPOCHS=150
BATCH_TRAIN=32
BATCH_INF=64

# Same DimeNet++ overrides you liked from earlier
MODEL_OVERRIDES="model.init.hidden_channels=128 model.init.num_blocks=4"

########################
# 2) Trainers to compare
########################

TRAINERS=(
  "semi-supervised-ensemble"
  "mean-teacher"
  "consistency-aug"
  "n-cps"
)

TRAINER=${TRAINERS[$((LSB_JOBINDEX - 1))]}

########################
# 3) W&B naming
########################

export WANDB_PROJECT="qm9_ssl"
export WANDB_GROUP="dimenetpp_ssl_methods"
export WANDB_NAME="${MODEL}_${TRAINER}_lr${LR}_wd${WD}_job${LSB_JOBINDEX}"

########################
# 4) Run
########################

set -x

python src/run.py \
  model=$MODEL \
  trainer=$TRAINER \
  trainer.train.total_epochs=$EPOCHS \
  trainer.train.validation_interval=10 \
  trainer.init.optimizer.lr=$LR \
  trainer.init.optimizer.weight_decay=$WD \
  dataset.init.batch_size_train=$BATCH_TRAIN \
  dataset.init.batch_size_inference=$BATCH_INF \
  $MODEL_OVERRIDES
