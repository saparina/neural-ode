from types import FunctionType
import tensorflow as tf
tfe = tf.contrib.eager


tf.register_tensor_conversion_function(
    FunctionType,
    lambda value, *args, **kwargs: tf.zeros(0))


def odeint_grad(grad_output, func, yt, t, args):
  ysize = yt.shape[1].value
  def backward_dynamics(state, t, args):
    funcstate = state[:ysize]
    adjoint = state[ysize:2 * ysize]
    fval = func(funcstate, t, args)
    grad = tfe.gradients_function(
        lambda *args: tf.tensordot(func(*args), -adjoint, 1))
    state_grad, t0_grad, adjoint_grad = grad(funcstate, t, args)
    return tf.concat([fval, state_grad, t0_grad[None], adjoint_grad], 0)

  y_grad = grad_output[-1]
  t0_grad = 0
  time_grads = []
  args_grad = tf.zeros_like(args)
  for i in range(ysize - 1, 0, -1):
    time_grads.append(
        tf.tensordot(func(yt[i], t[i], args), grad_output[i], 1))
    t0_grad = t0_grad - time_grads[-1]

    rev_t = tf.convert_to_tensor([t[i], t[i - 1]])
    backward_state = tf.concat([yt[i], y_grad, t0_grad[None], args_grad], 0)
    _t = -rev_t
    _base_reverse_f = backward_dynamics
    fc = lambda y, t, args: [-f_ for f_ in _base_reverse_f(y,-t, args)]

    backward_answer = tf.contrib.integrate.odeint(
        lambda y0, t: fc(y0, t, args),
        y0=backward_state,
        t=_t)[-1]
    _, y_grad, t_grad, args_grad = tf.split(backward_answer,
                                            [ysize, ysize, 1, ysize])
    y_grad = y_grad + grad_output[i - 1, :]
  time_grads.append(t0_grad)
  time_grads.reverse()
  time_grads = tf.convert_to_tensor(time_grads)
  return None, y_grad, time_grads, args_grad


@tf.custom_gradient
def odeint(func, y0, t, args):
  yt = tf.contrib.integrate.odeint(lambda y0, t: func(y0, t, args), y0, t)
  return yt, lambda grad_output: odeint_grad(grad_output, func, yt, t, args)
