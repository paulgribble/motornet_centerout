#!/bin/zsh
# loss weights from Shabazi et al 2024 A Context-Free Model of Savings in Motor Learning 10.1101/2025.03.26.645562

python go.py \
  --n_batch=5000 \
  --batch_size=64 \
  --interval=200 \
  --catch_trial_perc=50 \
  --n_models=3 \
  --dir_name=models_shahbazi \
  --loss_weight_position=1e+3 \
  --loss_weight_speed=0 \
  --loss_weight_jerk=1e+5 \
  --loss_weight_muscle=1e-1 \
  --loss_weight_muscle_derivative=0 \
  --loss_weight_hidden=1e-5 \
  --loss_weight_hidden_derivative=0


