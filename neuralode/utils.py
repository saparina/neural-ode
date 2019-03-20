import numpy as np
import torch


def _check_none_zero(tensor, shape, device):
    return tensor.to(device) if tensor is not None else torch.zeros(*shape).to(device)


def euler(z0, t0, t1, f, eps=0.05):
    n_steps = np.round((torch.abs(t1 - t0)/eps).max().item())
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
    for i_step in range(int(n_steps)):
        z = z + h * f(z, t)
        t = t + h
    return z


def runge_kutta(z0, t0, t1, f, eps=0.001):
    n_steps = np.round((torch.abs(t1 - t0)/(eps * 4)).max().item())
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
    for i_step in range(int(n_steps)):
        k1 = f(z, t)
        k2 = f(z + h/2 * k1, t + h/2)
        k3 = f(z + h/2 * k2, t + h/2)
        k4 = f(z + h * k3, t + h)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return z
