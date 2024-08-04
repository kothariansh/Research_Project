#!/bin/bash

# Training commands
echo "TRAINING TSP50 MODEL"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 131 --checkpoint_epochs 10 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --edit_fn global_perturb --hardness_adaptive_percent 20 --ewc_lambda 1 --run_name test_ewc_20batch_1_perturb_dcd
