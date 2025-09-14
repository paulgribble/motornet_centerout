#!/bin/zsh
# loss weights from Kashefi 2025 Compositional neural dynamics during reaching

python go.py \
  --n_batch=5000 \
  --batch_size=64 \
  --interval=200 \
  --catch_trial_perc=50 \
  --n_models=3 \
  --dir_name=models_kashefi \
  --loss_weight_position=1e+0 \
  --loss_weight_speed=1e-3 \
  --loss_weight_jerk=1e-4 \
  --loss_weight_muscle=1e-4 \
  --loss_weight_muscle_derivative=1e-4 \
  --loss_weight_hidden=1e-2 \
  --loss_weight_hidden_derivative=1e-1


