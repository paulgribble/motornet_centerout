import os
import json
import numpy as np
import torch as th
import motornet as mn
from tqdm import tqdm
import pickle

from joblib import Parallel, delayed
import multiprocessing

from my_policy import Policy  # the RNN
from my_task import CentreOutFF  # the task
from my_loss import cal_loss  # the loss function
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

import matplotlib.pyplot as plt

model_name = "m0"
cfg_file = "models/" + model_name + "/" + model_name + "_cfg.json"
weight_file = "models/" + model_name + "/" + model_name + "_weights"
batch_size, catch_trial_perc, condition, ff_coefficient, detach = 8, 0, 'test', 0, True
device = th.device("cpu")

# load configuration
cfg = json.load(open(cfg_file, 'r'))

if ff_coefficient is None:
    ff_coefficient = cfg['ff_coefficient']

# environment
name = cfg['name']
# effector
muscle_name = cfg['effector']['muscle']['name']
timestep = cfg['effector']['dt']
muscle = getattr(mn.muscle, muscle_name)()
effector = mn.effector.RigidTendonArm26(muscle=muscle, timestep=timestep)
# delay
proprioception_delay = cfg['proprioception_delay']*cfg['dt']
vision_delay = cfg['vision_delay']*cfg['dt']
# noise
action_noise = cfg['action_noise'][0]
proprioception_noise = cfg['proprioception_noise'][0]
vision_noise = cfg['vision_noise'][0]
# initialize environment
max_ep_duration = cfg['max_ep_duration']
env = CentreOutFF(effector=effector, max_ep_duration=max_ep_duration, name=name,
                    action_noise=action_noise, proprioception_noise=proprioception_noise,
                    vision_noise=vision_noise, proprioception_delay=proprioception_delay,
                    vision_delay=vision_delay)

# network
w = th.load(weight_file, weights_only=True)
num_hidden = int(w['gru.weight_ih_l0'].shape[0]/3)
if 'h0' in w.keys():
    policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=True)
else:
    policy = Policy( env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=False)
policy.load_state_dict(w)

h = policy.init_hidden(batch_size=batch_size)
obs, info = env.reset(condition=condition,
                        catch_trial_perc=catch_trial_perc,
                        ff_coefficient=ff_coefficient,
                        options={'batch_size': batch_size})
terminated = False

# Initialize a dictionary to store lists
data = {
    'xy': [],
    'obs': [],
    'tg': [],
    'vel': [],
    'all_actions': [],
    'all_hidden': [],
    'all_muscle': [],
    'all_force': [],
}

while not terminated:
    # Append data to respective lists
    data['all_hidden'].append(h[0, :, None, :])
    data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])

    action, h = policy(obs, h)
    obs, _, terminated, _, info = env.step(action=action)

    data['xy'].append(info["states"]["fingertip"][:, None, :])
    data['obs'].append(obs[:, None, :])
    data['tg'].append(info["goal"][:, None, :])
    data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
    data['all_actions'].append(action[:, None, :])
    data['all_force'].append(info['states']['muscle'][:, 6, None, :])

# Concatenate the lists
for key in data:
    data[key] = th.cat(data[key], axis=1)

if detach:
    # Detach tensors if needed
    for key in data:
        data[key] = th.detach(data[key])


plt.plot(data['tg'][:,:,0].T)
plt.plot(data['tg'][:,:,1].T)
plt.plot(data['obs'][:,:,0].T,'--')
plt.plot(data['obs'][:,:,1].T,'--')
plt.show()
