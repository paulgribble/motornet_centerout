#!/bin/zsh

python go.py \
  --n_batch=5000 \
  --batch_size=64 \
  --interval=100 \
  --catch_trial_perc=40 \
  --n_models=5 \
  --dir_name=models \
  --loss_weight_position=1e+3 \
  --loss_weight_speed=5e+2 \
  --loss_weight_jerk=1e+4 \
  --loss_weight_muscle=0 \
  --loss_weight_muscle_derivative=0 \
  --loss_weight_hidden=0 \
  --loss_weight_hidden_derivative=0


