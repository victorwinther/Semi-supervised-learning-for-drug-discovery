#!/bin/bash
#BSUB -q c02516
#BSUB -J "qm9_consistency_hp[1-6]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/qm9_consistency_hp_%J_%I.out
#BSUB -e err/qm9_consistency_hp_%J_%I.err

source ../DeepLearning/venv/bin/activate


BASE_LR=0.001
BASE_WD=0.0001
RAMP=50


case "$LSB_JOBINDEX" in
  1)
    CONS_WEIGHT=1.0
    NOISE_STD=0.05
    MASK_PROB=0.1
    ;;
  2)
    CONS_WEIGHT=5.0
    NOISE_STD=0.05
    MASK_PROB=0.1
    ;;
  3)
    CONS_WEIGHT=1.0
    NOISE_STD=0.1
    MASK_PROB=0.1
    ;;
  4)
    CONS_WEIGHT=1.0
    NOISE_STD=0.05
    MASK_PROB=0.25
    ;;
  5)
    CONS_WEIGHT=1.0
    NOISE_STD=0.15
    MASK_PROB=0.1
    ;;
  6)
    CONS_WEIGHT=5.0
    NOISE_STD=0.1
    MASK_PROB=0.1
    ;;
esac

python src/run.py \
  model=dimenetpp \
  trainer=consistency-aug \
  trainer.train.total_epochs=80 \
  trainer.init.optimizer.lr=$BASE_LR \
  trainer.init.optimizer.weight_decay=$BASE_WD \
  trainer.init.consistency_weight=$CONS_WEIGHT \
  trainer.init.aug_noise_std=$NOISE_STD \
  trainer.init.aug_mask_prob=$MASK_PROB \
  trainer.init.ramp_up_epochs=$RAMP \
  model.init.hidden_channels=128 \
  dataset.init.batch_size_train=32 \
  dataset.init.batch_size_inference=64