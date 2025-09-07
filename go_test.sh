#!/bin/zsh

python go.py \
  --n_batch=50 \
  --batch_size=32 \
  --interval=100 \
  --catch_trial_perc=50 \
  --n_models=4 \
  --dir_name=models
