#!/bin/bash
#BSUB -q c02516
#BSUB -J "qm9_dimenet_mt_opt[1-6]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/qm9_dimenet_mt_opt_%J_%I.out
#BSUB -e err/qm9_dimenet_mt_opt_%J_%I.err

# source ~/.venvs/ssl/bin/activate

case "$LSB_JOBINDEX" in
  1)
    LR=0.0005
    WD=0.0
    ;;
  2)
    LR=0.0005
    WD=0.0001
    ;;
  3)
    LR=0.0005
    WD=0.0005
    ;;
  4)
    LR=0.001
    WD=0.0
    ;;
  5)
    LR=0.001
    WD=0.0001
    ;;
  6)
    LR=0.001
    WD=0.0005
    ;;
esac

python src/run.py \
  model=dimenetpp \
  trainer=mean-teacher \
  trainer.train.total_epochs=40 \
  trainer.init.optimizer.lr=$LR \
  trainer.init.optimizer.weight_decay=$WD \
  model.init.hidden_channels=128 \
  model.init.num_blocks=4 \
  dataset.init.batch_size_train=32 \
  dataset.init.batch_size_inference=64
