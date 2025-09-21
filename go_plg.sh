#!/bin/zsh
#

python go.py \
  --n_batch=10000 \
  --batch_size=128 \
  --interval=1000 \
  --catch_trial_perc=50 \
  --n_models=5 \
  --dir_name=models \
  --loss_weight_position=1e+1 \
  --loss_weight_speed=1e-3 \
  --loss_weight_jerk=1e+3 \
  --loss_weight_muscle=5e-4 \
  --loss_weight_muscle_derivative=1e-3 \
  --loss_weight_hidden=1e-1 \
  --loss_weight_hidden_derivative=1e-0


