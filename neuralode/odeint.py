from types import FunctionType
import tensorflow as tf

from neuralode.utils import *
from neuralode.utils import _flatten_convert_none_to_zeros, _flatten

tfe = tf.contrib.eager


tf.register_tensor_conversion_function(
    FunctionType,
    lambda value, *args, **kwargs: tf.zeros(0))


def _flat(tensor):
  return tf.reshape(tensor, (-1,))


def odeint_grad(grad_output, func, yt, t, _args):
  yshape = yt.shape[1:]
  print(yt.shape)
  ysize = tf.reduce_prod(yshape)
  # print(ysize, args[0].shape)
  args = _flatten(_args)

  def backward_dynamics(state, t):
    y = tf.reshape(state[:ysize], yshape)
    adjoint_grad_y = state[ysize:2 * ysize]

    with tf.GradientTape() as tape:
        tape.watch(t)
        tape.watch(y)
        fval = func(y, t)
    vjp_t, vjp_y, vjp_params = tape.gradient(fval, [t, y, f_params])

    vjp_t = (tf.zeros_like(t, dtype=t.dtype),) if vjp_t is None else vjp_t

    vjp_params = tf.zeros_like(f_params, dtype=f_params.dtype) if vjp_params is None else vjp_params

    vjp_y = tuple(tf.zeros_like(y_, dtype=y_.dtype)
                  if vjp_y_ is None else vjp_y_
                  for vjp_y_, y_ in zip(vjp_y, y))
    vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

    if len(f_params) == 0:
        vjp_params = tf.convert_to_tensor(0., dtype=vjp_y.dype)
    return tf.concat([_flatten(fval), *vjp_y, vjp_t, vjp_params], 0)

  y_grad = grad_output[-1]
  t0_grad = 0
  time_grads = []
  flat_args = tf.concat([_flat(arg) for arg in args], 0)
  args_grad = tf.zeros_like(flat_args)
  for i in range(yt.shape[0].value - 1, 0, -1):
    new_t_grad = tf.tensordot(_flat(func(yt[i], t[i], args)),
                              _flat(grad_output[i]), 1)
    time_grads.append(new_t_grad)
    t0_grad = t0_grad - new_t_grad

    rev_t = tf.convert_to_tensor([t[i], t[i - 1]])
    backward_state = tf.concat([_flat(yt[i]), _flat(y_grad), t0_grad[None],
                                args_grad], 0)
    _t = -rev_t
    f_params = tuple(_args)
    fc = lambda y, t: [-val for val in backward_dynamics(y, -t)]

    backward_answer = tf.contrib.integrate.odeint(
        lambda y0, t: fc(y0, t),
        y0=backward_state,
        t=_t)[-1]
    _, y_grad, t_grad, args_grad = tf.split(backward_answer,
                                            [ysize, ysize, 1, -1])
    y_grad = y_grad + _flat(grad_output[i - 1, :])

  time_grads.append(t0_grad)
  time_grads.reverse()
  time_grads = tf.convert_to_tensor(time_grads)
  args_grad = tf.split(args_grad, [tf.reduce_prod(arg.shape) for arg in _args])
  args_grad = [tf.reshape(g, arg.shape) for g, arg in zip(args_grad, _args)]

  print(y_grad, time_grads, args_grad)

  return None, y_grad, time_grads, args_grad


@tf.custom_gradient
def odeint(func, y0, t):
  yt = tf.contrib.integrate.odeint(func, y0, t)
  def grad_fn(grad_output, variables=None):
    *grad_inputs, grad_variables = odeint_grad(grad_output, func, yt, t, variables)
    return grad_inputs, grad_variables

  print(yt, grad_fn) 
  return yt, grad_fn
