#!/bin/bash

python go.py \
  --n_batch=20000 \
  --batch_size=32 \
  --interval=1000 \
  --catch_trial_perc=50 \
  --n_models=10 \
  --dir_name=models
