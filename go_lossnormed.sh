#!/bin/zsh

python go.py \
  --n_batch=2000 \
  --n_units=256 \
  --batch_size=256 \
  --interval=100 \
  --catch_trial_perc=40 \
  --n_models=3 \
  --dir_name=models \
  --loss_weight_position=1e+2 \
  --loss_weight_speed=5e+0 \
  --loss_weight_jerk=1e+4 \
  --loss_weight_muscle=1e-2 \
  --loss_weight_muscle_derivative=1e-02 \
  --loss_weight_hidden=1e-01 \
  --loss_weight_hidden_derivative=1e+01


