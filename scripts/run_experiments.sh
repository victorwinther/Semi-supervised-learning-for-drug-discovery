#!/bin/bash
#BSUB -q c02516
#BSUB -J "gnn_ssl_sweep"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o out/gnn_ssl_sweep_%J_%I.out
#BSUB -e err/gnn_ssl_sweep_%J_%I.err


echo "Starting Phase 1: Architecture Search"

python src/run.py \
    model=gcn \
    trainer=semi-supervised-ensemble \
    dataset.batch_size_train=64 \
    logger.group="Phase1_Arch_Search" \
    logger.name="GCN_Supervised_Full" \
    dataset.splits=[110000,0,10000,10000] 

python src/run.py \
    model=dimenetpp \
    trainer=semi-supervised-ensemble \
    dataset.batch_size_train=32 \
    logger.group="Phase1_Arch_Search" \
    logger.name="DimeNetPP_Supervised_Full" \
    dataset.splits=[110000,0,10000,10000]

echo "Starting Phase 2: Low Data Baselines"

python src/run.py \
    model=dimenetpp \
    trainer=semi-supervised-ensemble \
    logger.group="Phase2_LowData_Baseline" \
    logger.name="DimeNet_1k_Labels" \
    dataset.splits=[1000,109000,10000,10000] 

python src/run.py \
    model=dimenetpp \
    trainer=semi-supervised-ensemble \
    logger.group="Phase2_LowData_Baseline" \
    logger.name="DimeNet_5k_Labels" \
    dataset.splits=[5000,105000,10000,10000]


echo "Starting Phase 3: SSL Methods"

for WEIGHT in 0.1 1.0 10.0
do
    python src/run.py \
        model=dimenetpp \
        trainer=mean_teacher \
        trainer.unsup_weight=$WEIGHT \
        logger.group="Phase3_SSL_MeanTeacher" \
        logger.name="MeanTeacher_w${WEIGHT}" \
        dataset.splits=[1000,109000,10000,10000]
done

for NOISE in 0.01 0.05
do
    python src/run.py \
        model=dimenetpp \
        trainer=consistency \
        trainer.aug_noise_std=$NOISE \
        logger.group="Phase3_SSL_Consistency" \
        logger.name="ConsAug_noise${NOISE}" \
        dataset.splits=[1000,109000,10000,10000]
done

python src/run.py \
    model=dimenetpp \
    trainer=ncps \
    logger.group="Phase3_SSL_NCPS" \
    logger.name="NCPS_Ensemble" \
    dataset.splits=[1000,109000,10000,10000]