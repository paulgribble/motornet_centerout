import os
import json
import numpy as np
import torch as th
import motornet as mn
import pickle

from my_policy import Policy  # the RNN
from my_task import CentreOutFF  # the task
from my_utils import (
    save_model,
    print_losses,
    plot_stuff,
    run_episode,
    test,
    plot_training_log,
    plot_simulations,
    plot_activation,
    plot_kinematics,
)  # utility functions


model_name = 'm0'
batch = 'current'

# load saved command line arguments
cmd_args_file = "models/" + model_name + "/" + "cmd_args.json"
if os.path.exists(cmd_args_file):
    with open(cmd_args_file, 'r') as f:
        cmd_args = json.load(f)
    loss_weights = cmd_args.get('loss_weights', None)
    print(f"Loaded parameters: n_units={cmd_args.get('n_units', 'not saved')}, loss_weights={loss_weights}")
else:
    loss_weights = None
    print(f"No cmd_args.json found, using default loss weights")

# run model tests and make plots
data, _ = test(
        "models/" + model_name + "/" + "cfg.json",
        "models/" + model_name + "/" + "weights",
        loss_weights=loss_weights,
)
plot_stuff(data, "models/" + model_name + "/", batch=batch)
