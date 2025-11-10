#!/bin/bash
#BSUB -q c02516
#BSUB -J "qm9_hp[1-6]"         # 4 jobs in the array
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/qm9_hp_%J_%I.out  # %I = array index
#BSUB -e err/qm9_hp_%J_%I.err

# source ~/.venvs/ssl/bin/activate


# Choose hyperparams based on LSB_JOBINDEX
case "$LSB_JOBINDEX" in
  1)
    LR=0.0005
    HIDDEN=64
    NBLOCKS=3
    ;;
  2)
    LR=0.0005
    HIDDEN=128
    NBLOCKS=3
    ;;
  3)
    LR=0.001
    HIDDEN=64
    NBLOCKS=3
    ;;
  4)
    LR=0.001
    HIDDEN=128
    NBLOCKS=3
    ;;
  5)
    LR=0.001
    HIDDEN=64
    NBLOCKS=4
    ;;
  6)
    LR=0.001
    HIDDEN=128
    NBLOCKS=4
    ;;
esac

python src/run.py \
  model=dimenetpp \
  trainer.train.total_epochs=40 \
  trainer.init.optimizer.lr=$LR \
  model.init.hidden_channels=$HIDDEN \
  model.init.num_blocks=$NBLOCKS \
  dataset.init.batch_size_train=32 \
  dataset.init.batch_size_inference=64