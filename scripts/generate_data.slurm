#!/bin/bash

# Generate Data
python generate_data.py --problem tsp --name test --seed 1234

# Eval commands
echo "STARTING UNIF EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_unif50_test_seed1234.pkl
echo "STARTING GMM EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_gmm50_test_seed1234.pkl
echo "STARTING POLY EVAL"
python concorde_baseline.py --data_path data/tsp/tsp_poly50_test_seed1234.pkl
