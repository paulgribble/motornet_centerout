#!/bin/zsh

python go.py \
  --n_units=256 \
  --n_batch=10000 \
  --batch_size=64 \
  --interval=100 \
  --catch_trial_perc=37.5 \
  --n_models=10 \
  --loss_weight_position=1e+0 \
  --loss_weight_speed=1e-3 \
  --loss_weight_jerk=1e-4 \
  --loss_weight_muscle=1e-4 \
  --loss_weight_muscle_derivative=1e-4 \
  --loss_weight_hidden=1e-2 \
  --loss_weight_hidden_derivative=1e-1 \
  --dir_name=models_plg

