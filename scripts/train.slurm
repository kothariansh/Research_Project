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
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 16384 --batch_size 1024 --n_epochs 40 --checkpoint_epochs 10 --run_name test_hac
python run.py --problem tsp --graph_size 50 --baseline rollout --edit_fn local_perturb --hardness_adaptive --epoch_size 16384 --batch_size 1024 --n_epochs 40 --checkpoint_epochs 10 --run_name test_dcd_hac
