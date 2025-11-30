#!/bin/bash
#BSUB -q c02516
# 5 models × 6 (LR,WD) combos = 30 jobs
#BSUB -J "qm9_arch_sweep[1-30]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/qm9_arch_sweep_%J_%I.out
#BSUB -e err/qm9_arch_sweep_%J_%I.err

# Optional: activate your venv
# source ~/.venvs/ssl/bin/activate

########################
# 1) Define search grid
########################

MODELS=(gcn gin schnet dimenetpp visnet attentivefp)

LRS=(0.0005 0.0005 0.0005 0.001 0.001 0.001)
WDS=(0.0    0.0001 0.0005 0.0   0.0001 0.0005)
NUM_COMBOS=${#LRS[@]}

TRAINER="semi-supervised-ensemble"

EPOCHS=150
BATCH_TRAIN=32
BATCH_INF=64

###########################################
# 2) Map LSB_JOBINDEX → (model, LR, WD)
###########################################

IDX=$((LSB_JOBINDEX - 1))
MODEL_IDX=$(( IDX / NUM_COMBOS ))
COMBO_IDX=$(( IDX % NUM_COMBOS ))

MODEL=${MODELS[$MODEL_IDX]}
LR=${LRS[$COMBO_IDX]}
WD=${WDS[$COMBO_IDX]}

###########################################
# 3) Optional per-model hyperparams
###########################################

MODEL_OVERRIDES=""

case "$MODEL" in
  gcn)
    MODEL_OVERRIDES="model.init.hidden_channels=256 model.init.num_layers=4"
    ;;
  gin)
    MODEL_OVERRIDES="model.init.hidden_channels=256 model.init.num_layers=5"
    ;;
  schnet)
    MODEL_OVERRIDES="model.init.hidden_channels=128 model.init.num_interactions=6"
    ;;
  dimenetpp)
    MODEL_OVERRIDES="model.init.hidden_channels=128 model.init.num_blocks=4"
    ;;
  visnet)
    MODEL_OVERRIDES="model.init.hidden_channels=128"
    ;;
esac

###########################################
# 4) W&B grouping via Hydra overrides
###########################################

GROUP="arch_sweep_qm9"
PROJECT="semi-supervised-learning-for-drug-discovery"   # or "qm9_ssl" if you prefer

RUN_NAME="${MODEL}_${TRAINER}_lr${LR}_wd${WD}_job${LSB_JOBINDEX}"

###########################################
# 5) Run
###########################################

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
  logger.group=$GROUP \
  logger.project_name=$PROJECT \
  logger.name="$RUN_NAME" \
  $MODEL_OVERRIDES
