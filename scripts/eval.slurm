#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --account=cse
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00

#SBATCH --output=out_eval.txt
#SBATCH --error=err_eval.txt



# Setup commands
. ~/.bashrc
cd /gscratch/socialrl/ml10872/tsp/Dual-Curriculum-TSP/
conda activate dcd_tsp

# Evaluation commands
echo "TSP20 dcd_local without local search"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP20 baseline without local search"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP50 dcd_local with local search"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP50 baseline with local search"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
