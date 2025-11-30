#!/bin/bash
#BSUB -q c02516
# Grid: 4 weights × 3 decays = 12 jobs
#BSUB -J "qm9_ssl_sweep[1-12]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/ssl_sweep_%J_%I.out
#BSUB -e err/ssl_sweep_%J_%I.err

# source ~/.venvs/ssl/bin/activate

########################
# 1) Define Search Grid
########################

# Fixed settings from Stage 1 (Best Architecture)
MODEL="dimenetpp"
BEST_LR=0.001
BEST_WD=0.0005

# SSL Hyperparameters to sweep
# unsup_weight: How much the consistency loss matters vs supervised loss
WEIGHTS=(0.01 0.1 1.0 5.0)

# ema_decay: How "slow" the teacher updates (0.999 is slower/stabler than 0.99)
DECAYS=(0.99 0.995 0.999)

NUM_WEIGHTS=${#WEIGHTS[@]}
NUM_DECAYS=${#DECAYS[@]}

TRAINER="mean-teacher"
EPOCHS=250   # SSL often needs longer to converge
RAMP_UP=50   # Usually good to keep fixed relative to total epochs

###########################################
# 2) Map LSB_JOBINDEX -> (unsup_weight, ema_decay)
###########################################

IDX=$((LSB_JOBINDEX - 1))

# Logic to iterate: Inner loop is DECAYS, Outer loop is WEIGHTS
WEIGHT_IDX=$(( IDX / NUM_DECAYS ))
DECAY_IDX=$(( IDX % NUM_DECAYS ))

UNSUP_WEIGHT=${WEIGHTS[$WEIGHT_IDX]}
EMA_DECAY=${DECAYS[$DECAY_IDX]}

###########################################
# 3) W&B Grouping
###########################################

GROUP="mean_teacher_hp_tuning"
PROJECT="semi-supervised-learning-for-drug-discovery"

RUN_NAME="${MODEL}_mt_w${UNSUP_WEIGHT}_d${EMA_DECAY}"

###########################################
# 4) Run
###########################################

echo "Running Job $LSB_JOBINDEX: Weight=$UNSUP_WEIGHT, Decay=$EMA_DECAY"

python src/run.py \
  model=$MODEL \
  trainer=$TRAINER \
  trainer.train.total_epochs=$EPOCHS \
  trainer.train.validation_interval=10 \
  trainer.init.optimizer.lr=$BEST_LR \
  trainer.init.optimizer.weight_decay=$BEST_WD \
  trainer.init.unsup_weight=$UNSUP_WEIGHT \
  trainer.init.ema_decay=$EMA_DECAY \
  trainer.init.ramp_up_epochs=$RAMP_UP \
  dataset.init.batch_size_train=32 \
  dataset.init.batch_size_inference=64 \
  logger.group=$GROUP \
  logger.project_name=$PROJECT \
  logger.name="$RUN_NAME" \
  model.init.hidden_channels=128 \
  model.init.num_blocks=4