import numpy as np
import tensorflow as tf
from neuralode.odeint import odeint
tf.enable_eager_execution()
tfe = tf.contrib.eager


def func(y, t, ampl):
  return ampl * tf.cos(tf.cast(y + 2 * np.pi * t, tf.float32))


y0 = np.array([0.,2], np.float32)
t = tf.linspace(0., 1., 10)
ampl = np.array([12,3], np.float32)

yt = odeint(func, y0, t, ampl)
print(yt.shape)


def odeint_func(ampl): return odeint(func, y0, t, ampl)

grad = tfe.gradients_function(odeint_func)
print(grad(np.array([12.,1], np.float32)))
