#!/bin/zsh
#

python go.py \
  --n_units=256 \
  --n_batch=20000 \
  --batch_size=64 \
  --interval=1000 \
  --catch_trial_perc=50 \
  --n_models=5 \
  --dir_name=models \
  --loss_weight_position=1e+2 \
  --loss_weight_speed=1e-1 \
  --loss_weight_jerk=1e-2 \
  --loss_weight_muscle=1e-2 \
  --loss_weight_muscle_derivative=1e-2 \
  --loss_weight_hidden=1e-0 \
  --loss_weight_hidden_derivative=1e+1

