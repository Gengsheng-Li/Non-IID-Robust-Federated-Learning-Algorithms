#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --nodelist gpu01
#SBATCH -o slurm_log/slurm%A_mix.log
#SBATCH -e slurm_log/slurm%A_mix.log

cd /data0/user/gsli/Non-IID-Robust-Federated-Learning-Algorithms/predict_brain_age_via_fl
conda activate FLlearner

# Train
python train_brainage_predictor.py \
    --user "gsli" \
    --agg_mth "FA" \
    --name_experiment "multi_site" \
    --data_dir  "../../../data/OpenBHB" \
    --num_clients 58 \
    --num_selected 10 \
    --num_rounds 100 \
    --epochs 5 \
    --lr 1e-2 \
    --batch_size 32 \
    --baseline_num 100 \
    --retrain_epochs 5 \
    # --retrain \
    # --verbose \
    # --forbid_wandb