#!/bin/zsh

# loss weights from Michaels 2025 Nature paper
# Sensory expectations shape neural population dynamics in motor circuits

python go.py \
  --n_batch=5000 \
  --batch_size=64 \
  --interval=200 \
  --catch_trial_perc=50 \
  --n_models=4 \
  --dir_name=models \
  --loss_weight_position=1e+3 \
  --loss_weight_speed=2e+2 \
  --loss_weight_jerk=1e+6 \
  --loss_weight_muscle=1e+0 \
  --loss_weight_muscle_derivative=0 \
  --loss_weight_hidden=1e-1 \
  --loss_weight_hidden_derivative=1e+4
