#!/bin/bash
#BSUB -q c02516
# Grid: 4 weights × 3 decays × 2 aug settings = 24 jobs
#BSUB -J "mt_sweep_komnulidt[1-24]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 4:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/mt_sweep_%J_%I.out
#BSUB -e err/mt_sweep_%J_%I.err

# Environment optimization (FIXED: match 4 cores)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4      # Changed from 8 to 4
export MKL_NUM_THREADS=4      # Changed from 8 to 4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

########################
# 1) Define Search Grid
########################

MODEL="dimenetpp"
TRAINER="mean-teacher"

# Fixed settings (from your previous runs)
BEST_LR=0.001
BEST_WD=0.005

# Fast experimentation settings
SUBSET_SIZE=10000        # Fast training (~1.5s/it)
EPOCHS=200              # Enough to see convergence
VAL_INTERVAL=10
BATCH_SIZE=128           # Changed from 256 to prevent OOM
NUM_WORKERS=4            # Matches available cores

# SSL Hyperparameters to sweep
# unsup_weight: How much consistency loss matters
WEIGHTS=(0.1 0.5 1.0 2.0)

# ema_decay: Teacher update speed (higher = slower/more stable)
DECAYS=(0.99 0.995 0.999)

# data_augmentation: Enable/disable coordinate noise
AUG_SETTINGS=(false true)

NUM_WEIGHTS=${#WEIGHTS[@]}
NUM_DECAYS=${#DECAYS[@]}
NUM_AUG=${#AUG_SETTINGS[@]}

# Ramp up: linear increase of unsup_weight over first N epochs
RAMP_UP=50

###########################################
# 2) Map LSB_JOBINDEX -> (weight, decay, aug)
###########################################

IDX=$((LSB_JOBINDEX - 1))

# 3D grid indexing: weight -> decay -> aug
AUG_IDX=$(( IDX % NUM_AUG ))
TEMP=$(( IDX / NUM_AUG ))
DECAY_IDX=$(( TEMP % NUM_DECAYS ))
WEIGHT_IDX=$(( TEMP / NUM_DECAYS ))

UNSUP_WEIGHT=${WEIGHTS[$WEIGHT_IDX]}
EMA_DECAY=${DECAYS[$DECAY_IDX]}
DATA_AUG=${AUG_SETTINGS[$AUG_IDX]}

###########################################
# 3) W&B Grouping
###########################################

GROUP="mean_teacher_sweep_fast"
PROJECT="semi-supervised-learning-for-drug-discovery"

if [ "$DATA_AUG" = "true" ]; then
    AUG_SUFFIX="_aug"
else
    AUG_SUFFIX=""
fi

RUN_NAME="${MODEL}_w${UNSUP_WEIGHT}_d${EMA_DECAY}${AUG_SUFFIX}"

###########################################
# 4) Run Training
###########################################

echo "=========================================="
echo "Job $LSB_JOBINDEX: Starting sweep run"
echo "Weight: $UNSUP_WEIGHT"
echo "EMA Decay: $EMA_DECAY"
echo "Data Aug: $DATA_AUG"
echo "Run Name: $RUN_NAME"
echo "=========================================="

python src/run.py \
  model=$MODEL \
  trainer=$TRAINER \
  trainer.train.total_epochs=$EPOCHS \
  trainer.train.validation_interval=$VAL_INTERVAL \
  trainer.init.optimizer.lr=$BEST_LR \
  trainer.init.optimizer.weight_decay=$BEST_WD \
  trainer.init.unsup_weight=$UNSUP_WEIGHT \
  trainer.init.ema_decay=$EMA_DECAY \
  trainer.init.ramp_up_epochs=$RAMP_UP \
  dataset.init.batch_size_train=$BATCH_SIZE \
  dataset.init.batch_size_inference=128 \
  dataset.init.num_workers=$NUM_WORKERS \
  dataset.init.subset_size=$SUBSET_SIZE \
  trainer.init.augment_coords=$DATA_AUG \
  trainer.init.coord_noise_std=0.05 \
  logger.group=$GROUP \
  logger.project_name=$PROJECT \
  logger.name="$RUN_NAME" \
  model.init.hidden_channels=128 \
  model.init.num_blocks=4

echo "Job $LSB_JOBINDEX completed: $RUN_NAME"