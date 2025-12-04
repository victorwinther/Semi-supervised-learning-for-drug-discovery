#!/bin/bash
### -- Queue: V100 Queue --
#BSUB -q gpul40s
### -- Job Array: 16 Jobs --
#BSUB -J "dime_opt[1-16]"
### -- Resources: 8 Cores (Requested), 1 GPU --
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
### -- Memory: 32GB RAM for 8 workers --
#BSUB -R "rusage[mem=16GB]"
### -- Time: 24h --
#BSUB -W 12:00
### -- Logs --
#BSUB -o out/opt_%J_%I.out
#BSUB -e err/opt_%J_%I.err

echo "------------------------------------------------"
echo "Job $LSB_JOBINDEX on node $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "------------------------------------------------"

module load cuda/11.8
export OMP_NUM_THREADS=8


if (( ($LSB_JOBINDEX - 1) / 8 % 2 == 0 )); then NOISE=0.01; else NOISE=0.02; fi

if (( ($LSB_JOBINDEX - 1) / 4 % 2 == 0 )); then WEIGHT=2.0; else WEIGHT=5.0; fi

if (( ($LSB_JOBINDEX - 1) / 2 % 2 == 0 )); then GAMMA=0.98; else GAMMA=0.99; fi

if (( ($LSB_JOBINDEX - 1) % 2 == 0 ));     then WD=0.001;  else WD=0.0001; fi

RUN_NAME="dim_n${NOISE}_w${WEIGHT}_g${GAMMA}_wd${WD}"

echo "Config: Noise=$NOISE | Weight=$WEIGHT | Gamma=$GAMMA | WD=$WD"
echo "Run Name: $RUN_NAME"

python src/run.py \
  model=dimenetpp trainer=mean-teacher\
  dataset.init.batch_size_train=256 \
  dataset.init.batch_size_inference=4096 \
  dataset.init.num_workers=8 \
  trainer.init.augment_coords=true \
  trainer.init.coord_noise_std=$NOISE \
  trainer.init.unsup_weight=$WEIGHT \
  trainer.init.ema_decay=0.999 \
  trainer.init.scheduler.gamma=$GAMMA \
  trainer.init.optimizer.weight_decay=$WD \
  trainer.train.total_epochs=250 \
  logger.group="overnight_sweep_v2" \
  logger.name="$RUN_NAME"