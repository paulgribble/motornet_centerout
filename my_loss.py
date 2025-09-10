import numpy as np
import torch as th


def cal_loss(data, loss_weights=None):

    loss = {'position': None,
            'speed': None,
            'jerk': None,
            'muscle': None,
            'muscle_derivative': None,
            'hidden': None,
            'hidden_derivative': None
            }

    loss['position']          = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1))
    loss['speed']             = th.mean(th.sum(th.square(data['vel']), dim=-1))
    loss['jerk']              = th.mean(th.sum(th.square(th.diff(data['vel'], n=2, dim=1)), dim=-1))
    loss['muscle']            = th.mean(th.sum(data['all_force'], dim=-1))
    loss['muscle_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_force'], n=1, dim=1)), dim=-1))
    loss['hidden']            = th.mean(th.sum(th.square(data['all_hidden']), dim=-1))
    loss['hidden_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], n=1, dim=1)), dim=-1))

    if (loss_weights is None):
        # from Kashefi et al 2025, Compositional neural dynamics during reaching
        # loss_weights = np.array([1e+0,   # position
        #                          1e-3,   # speed
        #                          1e-0,   # jerk 1e-4
        #                          1e-4,   # muscle
        #                          1e-4,   # muscle_derivative
        #                          1e-2,   # hidden
        #                          1e-1])  # hidden_derivative
        # from Shabazi et al 2024, 
        loss_weights = np.array([1e+3,   # position
                                    0,   # speed
                                 1e+5,   # jerk 1e-4
                                 1e-1,   # muscle
                                 3e-4,   # muscle_derivative
                                 1e-5,   # hidden
                                 1e-3])  # hidden_derivative

    losses_weighted = {
        'position':          loss_weights[0] * loss['position'],
        'speed_loss':        loss_weights[1] * loss['speed'],
        'jerk_loss':         loss_weights[2] * loss['jerk'],
        'muscle':            loss_weights[3] * loss['muscle'],
        'muscle_derivative': loss_weights[4] * loss['muscle_derivative'],
        'hidden':            loss_weights[5] * loss['hidden'],
        'hidden_derivative': loss_weights[6] * loss['hidden_derivative']
    }

    overall_loss = 0.0
    for loss in losses_weighted.keys():
        overall_loss += losses_weighted[loss]

    return overall_loss, losses_weighted

