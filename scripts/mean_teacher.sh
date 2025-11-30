#!/bin/bash
#BSUB -q c02516
#BSUB -J "qm9_dimenet_mt_hp[1-6]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/qm9_dimenet_mt_hp_%J_%I.out
#BSUB -e err/qm9_dimenet_mt_hp_%J_%I.err

# source ~/.venvs/ssl/bin/activate

# Fix these from your best run in step 2:
BASE_LR=0.0005
BASE_WD=0.0001

case "$LSB_JOBINDEX" in
  1)
    EMA=0.999
    UNSUP=0.05
    RAMP=50
    ;;
  2)
    EMA=0.999
    UNSUP=0.10   # your current setting
    RAMP=50
    ;;
  3)
    EMA=0.999
    UNSUP=0.20
    RAMP=50
    ;;
  4)
    EMA=0.995
    UNSUP=0.10
    RAMP=50
    ;;
  5)
    EMA=0.9995
    UNSUP=0.10
    RAMP=50
    ;;
  6)
    EMA=0.995
    UNSUP=0.20
    RAMP=100
    ;;
esac

python src/run.py \
  model=dimenetpp \
  trainer=mean-teacher \
  trainer.train.total_epochs=100 \
  trainer.init.optimizer.lr=$BASE_LR \
  trainer.init.optimizer.weight_decay=$BASE_WD \
  trainer.init.ema_decay=$EMA \
  trainer.init.unsup_weight=$UNSUP \
  trainer.init.ramp_up_epochs=$RAMP \
  model.init.hidden_channels=128 \
  model.init.num_blocks=4 \
  dataset.init.batch_size_train=32 \
  dataset.init.batch_size_inference=64
