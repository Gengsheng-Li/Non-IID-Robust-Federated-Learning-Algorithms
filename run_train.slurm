#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --nodelist gpu01
#SBATCH -o slurm_log/slurm%A_mix.log
#SBATCH -e slurm_log/slurm%A_mix.log

cd /data0/user/gsli/Non-IID-Robust-Federated-Learning-Algorithms
conda activate FLlearner

# Train
python train.py \
    --user "gsli" \
    --agg_mth "FA" \
    --real_world 2 \
    --num_clients 20 \
    --num_selected 6 \
    --classes_pc 6 \
    --baseline_num 100 \
    --num_rounds 150 \
    --epochs 5 \
    --lr 0.1 \
    --batch_size 32 \
    --retrain_epochs 5 \
    --verbose \
    # --retrain \