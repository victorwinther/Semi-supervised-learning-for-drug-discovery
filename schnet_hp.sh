#!/bin/bash
#BSUB -q c02516
#BSUB -J "qm9_schnet_hp[1-6]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/qm9_schnet_hp_%J_%I.out
#BSUB -e err/qm9_schnet_hp_%J_%I.err

# source ~/.venvs/ssl/bin/activate

# Choose hyperparams based on LSB_JOBINDEX
case "$LSB_JOBINDEX" in
  1)
    LR=0.0005
    HIDDEN=64
    NINTER=4
    ;;
  2)
    LR=0.0005
    HIDDEN=128
    NINTER=4
    ;;
  3)
    LR=0.0005
    HIDDEN=128
    NINTER=6
    ;;
  4)
    LR=0.001
    HIDDEN=128
    NINTER=6
    ;;
  5)
    LR=0.001
    HIDDEN=256
    NINTER=6
    ;;
  6)
    LR=0.001
    HIDDEN=128
    NINTER=8
    ;;
esac

source ../DeepLearning/venv/bin/activate

python src/run.py \
  model=schnet \
  trainer.train.total_epochs=200 \
  trainer.init.optimizer.lr=$LR \
  model.init.hidden_channels=$HIDDEN \
  model.init.num_filters=$HIDDEN \
  model.init.num_interactions=$NINTER \
  dataset.init.batch_size_train=32 \
  dataset.init.batch_size_inference=64
