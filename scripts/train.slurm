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
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 91 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 10 --ewc_fisher_n 128 --pretrain_path pretrained/tsp_50 --run_name ewc_10_smalln
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 91 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 10 --ewc_fisher_n 256 --ewc_from_unif --pretrain_path pretrained/tsp_50 --run_name ewc_10_unif_smalln

# Evaluation commands
echo "EVALUATING TSP50 ON UNIF AND GMM"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_unif_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_unif_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

echo "EVALUATING TSP50 ON POLY"
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/ewc_50_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
