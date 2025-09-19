#!/bin/zsh
# loss weights from Shabazi et al 2024 A Context-Free Model of Savings in Motor Learning 10.1101/2025.03.26.645562

python go.py \
  --n_batch=20000 \
  --batch_size=64 \
  --interval=1000 \
  --catch_trial_perc=50 \
  --n_models=5 \
  --dir_name=models \
  --loss_weight_position=1e+3 \
  --loss_weight_speed=1e-1 \
  --loss_weight_jerk=1e+5 \
  --loss_weight_muscle=1e-1 \
  --loss_weight_muscle_derivative=1e-1 \
  --loss_weight_hidden=1e-0 \
  --loss_weight_hidden_derivative=1e+1


