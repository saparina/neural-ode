import tensorflow as tf
from tensorflow.python.ops.gradient_checker_v2 import (compute_gradient,
                                                       max_error)
import numpy as np
from .odeint import odeint
tf.enable_eager_execution()


class TestOdeint(tf.test.TestCase):
  def test_sin(self):
    ampl = tf.Variable(1.)
    func = lambda x, t, ampl: ampl * tf.sin(x * tf.cast(t, tf.float32))
    y0 = np.array([0.46], np.float32)
    t = np.array([0.05, 0.14, 0.15, 0.61, 1.], np.float32)

    odeint_wrap = lambda ampl: odeint(lambda y0, t: func(y0, t, ampl), y0, t)
    *grads, = compute_gradient(odeint_wrap, [ampl])
    print(f"test sin error {max_error(*grads)}")
    self.assertLess(max_error(*grads), 1e-5)

  def test_log(self):
    func = lambda x, t, var: tf.log((tf.cast(x, t.dtype) + t)
                                    * tf.cast(var, t.dtype))
    y0 = np.array([0.659])
    t = np.linspace(0, 1, num=10)

    odeint_wrap = lambda var: odeint(lambda y0, t: func(y0, t, var), y0, t)
    var = tf.Variable(1., dtype=tf.float64)
    *grads, = compute_gradient(odeint_wrap, [var])
    print(f"test log error: {max_error(*grads)}")
    self.assertLess(max_error(*grads), 1e-5)
