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
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 10 --ewc_fisher_n 512 --pretrain_path pretrained/tsp_50 --run_name ewc_10_smalln
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 50 --ewc_fisher_n 512 --pretrain_path pretrained/tsp_50 --run_name ewc_50_smalln
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 0.1 --pretrain_path pretrained/tsp_50 --run_name ewc_01
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 500 --pretrain_path pretrained/tsp_50 --run_name ewc_500
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 10 --ewc_from_unif --pretrain_path pretrained/tsp_50 --run_name ewc_10_unif
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 50 --ewc_from_unif --pretrain_path pretrained/tsp_50 --run_name ewc_50_unif
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 100 --ewc_from_unif --pretrain_path pretrained/tsp_50 --run_name ewc_100_unif
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 200 --ewc_from_unif --pretrain_path pretrained/tsp_50 --run_name ewc_200_unif
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --edit_fn local_perturb --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 10 --pretrain_path pretrained/tsp_50 --run_name ewc_10_dcd
python run.py --problem tsp --graph_size 50 --baseline rollout --hardness_adaptive --edit_fn local_perturb --epoch_size 65536 --batch_size 1024 --n_epochs 101 --checkpoint_epochs 10 --lr_decay 0.98 --ewc_lambda 10 --ewc_from_unif --pretrain_path pretrained/tsp_50 --run_name ewc_10_unif_dcd


# Evaluation commands
echo "EVALUATING TSP50 ON UNIF AND GMM"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_50_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_50_smalln --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_01 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_01 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_500 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_500 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_50_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_50_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_100_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_100_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_200_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_200_unif --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_dcd --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_dcd --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_unif_dcd --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/ewc_10_unif_dcd --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

echo "EVALUATING TSP50 ON POLY"
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/ewc_10 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/ewc_50 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
