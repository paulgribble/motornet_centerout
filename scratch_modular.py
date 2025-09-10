import numpy as np
import torch as th
import motornet as mn

from my_policy_modular import ModularPolicyGRU  # the RNN
from my_task import CentreOutFF  # the task

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")

dt = 0.01    # time step in seconds
ep_dur = 1.0 # episode duration in seconds

mm = mn.muscle.RigidTendonHillMuscle() # muscle model
ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=dt) # effector model

# Initialize the environment
env = CentreOutFF(max_ep_duration=ep_dur, effector=ee,
                  proprioception_delay=0.01, vision_delay=0.07,
                  proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4)
obs, info = env.reset()

vision_mask = [1, 0, 0, 0]
proprio_mask = [0, 0, 0, 1]
task_mask = [1, 0, 0, 0]
connectivity_mask = np.array([[1, 0.2, 0, 0], [0.2, 1, 0.2, 0], [0, 0.2, 1, 0.5], [0, 0.2, 0, 1]])
connectivity_delay = np.zeros_like(connectivity_mask)
output_mask = [0, 0, 0, 1]
module_sizes = [64, 128, 64, 16]

spectral_scaling = 1
# input sparsity
vision_dim = np.arange(2)
proprio_dim = np.arange(12) + vision_dim[-1] + 1
task_dim = np.arange(3) + proprio_dim[-1] + 1
policy = ModularPolicyGRU(2+12+3, module_sizes, env.n_muscles,
                          vision_dim=vision_dim, proprio_dim=proprio_dim, task_dim=task_dim,
                          vision_mask=vision_mask, proprio_mask=proprio_mask, task_mask=task_mask,
                          connectivity_mask=connectivity_mask, output_mask=output_mask,
                          connectivity_delay=connectivity_delay,
                          proportion_excitatory=None, input_gain=1.,
                          spectral_scaling=spectral_scaling, device=device, activation='rect_tanh', output_delay=1)

optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

# TO DO:
#   env code should be basically fixed
#   task code should vary
#   see https://github.com/neural-control-and-computation-lab/MotorNet/tree/JAM-staging/MotorSaving


