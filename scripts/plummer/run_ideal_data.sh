#!/bin/bash

# Define variables
DATA_FILE_NAME="plummer_sphere_allstars.h5"

mkdir -p runs/allstars
cp options.json runs/allstars/

# Fit both DF and potential
# python ../fit_all.py \
#     --input $DATA_FILE_NAME \
#     --run-dir runs/allstars \
#     --basic-flow-benchmarking \
#     --basic-potential-benchmarking

# Fit only the potential
python ../fit_all.py \
    --input $DATA_FILE_NAME \
    --run-dir runs/allstars \
    --basic-potential-benchmarking \
    --no-flow-training \
    --no-flow-sampling