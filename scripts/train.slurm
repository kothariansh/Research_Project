#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=cse
#SBATCH --partition=ckpt
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
python run.py --problem tsp --graph_size 20 --baseline rollout --edit_fn local_perturb --epoch_size 16384 --batch_size 2048 --n_epochs 2500 --checkpoint_epochs 100 --run_name dcd_local
python run.py --problem tsp --graph_size 50 --baseline rollout --edit_fn local_perturb --epoch_size 16384 --batch_size 2048 --n_epochs 2500 --checkpoint_epochs 100 --run_name dcd_local
