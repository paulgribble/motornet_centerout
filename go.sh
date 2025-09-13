#!/bin/zsh

python go.py \
  --n_batch=5000 \
  --batch_size=64 \
  --interval=200 \
  --catch_trial_perc=50 \
  --n_models=4 \
  --n_units=256 \
  --dir_name=models 
