#!/bin/bash

#SBATCH --job-name=concorde
#SBATCH --account=cse
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gpus=0
#SBATCH --time=01-00:00:00

#SBATCH --output=out_concorde.txt
#SBATCH --error=err_concorde.txt



# Setup commands
. ~/.bashrc
cd /gscratch/socialrl/ml10872/tsp/Dual-Curriculum-TSP/
conda activate dcd_tsp

# Eval commands
echo "STARTING GMM EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_gmm50_test_seed1234.pkl
echo "STARTING UNIF EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_unif50_test_seed1234.pkl
