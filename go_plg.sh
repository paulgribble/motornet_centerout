#!/bin/zsh

python go.py \
  --n_units=128 \
  --n_batch=2000 \
  --batch_size=128 \
  --interval=100 \
  --catch_trial_perc=40 \
  --n_models=3 \
  --loss_weight_position=1e+1 \
  --loss_weight_speed=0 \
  --loss_weight_jerk=0 \
  --loss_weight_muscle=0 \
  --loss_weight_muscle_derivative=0 \
  --loss_weight_hidden=1e-3 \
  --loss_weight_hidden_derivative=1e+0 \
  --dir_name=models_plg

