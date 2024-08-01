#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --gpus=1
#SBATCH --time=02-00:00:00

#SBATCH --output=out.txt
#SBATCH --error=err.txt



# Setup commands
. ~/.bashrc
cd /gscratch/socialrl/ml10872/tsp/Dual-Curriculum-TSP/
conda activate dcd_tsp

# Training commands
echo "TRAINING TSP50"
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 71 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 50 --pretrain_path pretrained/tsp_50 --run_name ewc_50
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 71 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 500 --pretrain_path pretrained/tsp_50 --run_name ewc_500
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 71 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 1000 --pretrain_path pretrained/tsp_50 --run_name ewc_1000

# Evaluation commands
echo "EVALUATING TSP50 EWC"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_50 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_50 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_500 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_500 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_1000 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_1000 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
