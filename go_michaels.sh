#!/bin/zsh
# loss weights from Sensory expectations shape neural population dynamics in motor circuits

python go.py \
  --n_batch=10000 \
  --batch_size=64 \
  --interval=200 \
  --catch_trial_perc=50 \
  --n_models=3 \
  --dir_name=models_michaels \
  --loss_weight_position=1e+3 \
  --loss_weight_speed=2e+2 \
  --loss_weight_jerk=1e+5 \
  --loss_weight_muscle=1e+0 \
  --loss_weight_muscle_derivative=0 \
  --loss_weight_hidden=1e-1 \
  --loss_weight_hidden_derivative=1e+4


