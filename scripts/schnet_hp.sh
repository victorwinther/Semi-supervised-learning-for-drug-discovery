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

  case "$LSB_JOBINDEX" in
    1)
      LR=0.0005
      HIDDEN=128
      NINTER=6
      DROPOUT=0.0
      WD=0.005
      ;;
    2)
      LR=0.0005
      HIDDEN=128
      NINTER=6
      DROPOUT=0.1
      WD=0.005
      ;;
    3)
      LR=0.0005
      HIDDEN=128
      NINTER=6
      DROPOUT=0.25
      WD=0.005
      ;;
    4)
      LR=0.0005
      HIDDEN=128
      NINTER=6
      DROPOUT=0.35
      WD=0.005
      ;;
    5)
      LR=0.0005
      HIDDEN=128
      NINTER=6
      DROPOUT=0.1
      WD=0.01
      ;;
    6)
      LR=0.001
      HIDDEN=128
      NINTER=6
      DROPOUT=0.1
      WD=0.005
      ;;
  esac

  source ../DeepLearning/venv/bin/activate

  python src/run.py \
    model=schnet \
    trainer.train.total_epochs=150 \
    trainer.init.optimizer.lr=$LR \
    trainer.init.optimizer.weight_decay=$WD \
    model.init.hidden_channels=$HIDDEN \
    model.init.num_filters=$HIDDEN \
    model.init.num_interactions=$NINTER \
    model.init.dropout=$DROPOUT \
    model.init.add_mlp_head=true \
    model.init.mlp_hidden=128 \
    dataset.init.batch_size_train=32 \
    dataset.init.batch_size_inference=64
