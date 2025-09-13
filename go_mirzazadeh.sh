#!/bin/zsh
# loss weights from Mirzazadeh Poune in Jonathan's lab
# https://github.com/neural-control-and-computation-lab/MotorNet/tree/JAM-staging/MotorSaving

python go.py \
  --n_batch=5000 \
  --batch_size=64 \
  --interval=200 \
  --catch_trial_perc=50 \
  --n_models=4 \
  --dir_name=models_mirzazadeh \
--loss_weight_position=1e+3 \
--loss_weight_speed=0 \
--loss_weight_jerk=1e+3 \
--loss_weight_muscle=1e-1 \
--loss_weight_muscle_derivative=0 \
--loss_weight_hidden=0 \
--loss_weight_hidden_derivative=1e+4


