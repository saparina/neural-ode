from types import FunctionType
import tensorflow as tf

tfe = tf.contrib.eager


tf.register_tensor_conversion_function(
    FunctionType, lambda value, *args, **kwargs: tf.zeros(0))
tf.register_tensor_conversion_function(
    tf.keras.Model, lambda value, *args, **kwargs: tf.zeros(0))


def _flatten(tensors):
  if not isinstance(tensors, (list, tuple)): # given single tensor
    return tf.reshape(tensors, (-1,))
  return (tf.concat([tf.reshape(t, (-1,)) for t in tensors], 0)
          if tensors else tf.convert_to_tensor([]))


def _unflatten(tensor, shapes):
  split = tf.split(tensor, [tf.reduce_prod(s) for s in shapes])
  return [tf.reshape(t, s) for t, s in zip(split, shapes)]


def odeint_grad(grad_output, func, yt, t, variables=None,
                rtol=1e-6, atol=1e-12):
  yshape = yt.shape[1:]
  ysize = tf.reduce_prod(yshape)

  def backward_dynamics(state, t):
    y = tf.reshape(state[:ysize], yshape)
    adjoint_grad_y = tf.reshape(state[ysize:2 * ysize], yshape)

    with tf.GradientTape() as tape:
      tape.watch([t, y])
      fval = func(y, t)
    vjp_t, vjp_y, vjp_variables = tape.gradient(
        fval, [t, y, variables],
        output_gradients=-adjoint_grad_y)

    vjp_t = tf.zeros_like(t)
    vjp_y = _flatten(vjp_y)
    vjp_variables = _flatten(vjp_variables)

    # use negative values to integrate from t[i] to t[i - 1] using
    # tf.contrib.integrate.odeint.
    return -tf.concat([_flatten(fval), vjp_y, vjp_t[None], vjp_variables], 0)

  y_grad = grad_output[-1]
  t_grad = 0
  time_grads = []
  flat_variables = _flatten(variables)
  variables_grad = tf.zeros_like(flat_variables)
  for i in range(yt.shape[0].value - 1, 0, -1):
    new_t_grad = tf.tensordot(_flatten(func(yt[i], t[i])),
                              _flatten(grad_output[i]), 1)
    time_grads.append(float(new_t_grad))
    t_grad = float(t_grad - new_t_grad)

    backward_state = tf.concat([_flatten(yt[i]), _flatten(y_grad),
                                [t_grad], variables_grad], 0)
    backward_t = tf.convert_to_tensor([-t[i], -t[i - 1]])
    backward_answer = tf.contrib.integrate.odeint(
        backward_dynamics,
        y0=backward_state, t=backward_t,
        rtol=rtol, atol=atol)[-1]
    _, y_grad, t_grad, variables_grad = tf.split(backward_answer,
                                                 [ysize, ysize, 1, -1])
    y_grad = y_grad + _flatten(grad_output[i - 1, :])

  time_grads.append(float(t_grad))
  time_grads.reverse()
  time_grads = tf.convert_to_tensor(time_grads)

  if variables:
    variables_grad = _unflatten(variables_grad, [v.shape for v in variables])
  y_grad = tf.reshape(y_grad, yshape)
  return (tf.zeros(0), y_grad, time_grads), variables_grad


@tf.custom_gradient
def odeint(func, y0, t, rtol=1e-6, atol=1e-12):
  yt = tf.contrib.integrate.odeint(func, y0, t, rtol=rtol, atol=atol)

  def grad_fn(grad_output, variables=None):
    return odeint_grad(grad_output, func, yt, t, variables=variables,
                       rtol=rtol, atol=atol)
  return yt, grad_fn
