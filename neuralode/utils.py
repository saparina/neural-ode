import tensorflow as tf
import numpy as np
import torch


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        tf.reshape(p, [-1]) if p is not None else tf.reshape(tf.zeros_like(q), [-1])
        for p, q in zip(sequence, like_sequence)
    ]
    return tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])


def _flatten(sequence):
    flat = [tf.reshape(p, [-1]) for p in sequence]
    return tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])


def _check_none_zero(tensor, shape, device):
    return tensor.to(device) if tensor is not None else torch.zeros(*shape).to(device)


def euler(z0, t0, t1, f, eps=0.05):
    n_steps = np.round((np.abs(t1 - t0)/eps).max().item())
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
    # print(z0.size())
    # print(f(z0, t0).size())
    for i_step in range(int(n_steps)):
        z = z + h * f(z, t)
        t = t + h
    return z