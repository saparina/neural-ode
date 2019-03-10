from types import FunctionType
import tensorflow as tf
tfe = tf.contrib.eager


tf.register_tensor_conversion_function(
    FunctionType,
    lambda value, *args, **kwargs: tf.zeros(0))


def _flat(tensor):
  return tf.reshape(tensor, (-1,))


def odeint_grad(grad_output, func, yt, t, args):
  yshape = yt.shape[1:]
  ysize = tf.reduce_prod(yshape)

  def backward_dynamics(state, t, args):
    funcstate = tf.reshape(state[:ysize], yshape)
    adjoint = state[ysize:2 * ysize]
    # TODO: compute vjps with adjoint.
    return tf.concat([fval, state_grad, t0_grad[None], adjoint_grad], 0)

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
    fc = lambda y, t, args: [-val for val in backward_dynamics(y, -t, args)]

    backward_answer = tf.contrib.integrate.odeint(
        lambda y0, t: fc(y0, t, args),
        y0=backward_state,
        t=_t)[-1]
    _, y_grad, t_grad, args_grad = tf.split(backward_answer,
                                            [ysize, ysize, 1, -1])
    y_grad = y_grad + grad_output[i - 1, :]
  time_grads.append(t0_grad)
  time_grads.reverse()
  time_grads = tf.convert_to_tensor(time_grads)
  args_grad = tf.split(args_grad, [tf.reduce_prod(arg.shape) for arg in args])
  args_grad = [tf.reshape(g, arg.shape) for g, arg in zip(args_grad, args)]
  return None, y_grad, time_grads, args_grad


@tf.custom_gradient
def odeint(func, y0, t):
  yt = tf.contrib.integrate.odeint(func, y0, t)

  def grad_fn(grad_output, variables=None):
    _, y_grad, time_grads, var_grad = odeint_grad(
        grad_output, func, yt, t, variables)
    return None, y_grad, time_grads, var_grad

  return yt, grad_fn
