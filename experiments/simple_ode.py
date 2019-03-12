import numpy as np
import tensorflow as tf
from neuralode.odeint import odeint
from neuralode.utils import _flatten

tf.enable_eager_execution()
tfe = tf.contrib.eager


data_size = 7
batch_time = 2
batch_size = 5

true_y0 = np.array([[0.5, 0.3]], dtype=np.float32)
true_A = np.array([[-0.1, -1.], [1., -0.1]],  dtype=np.float32)
t = tf.linspace(0., 10., data_size)


class Model(tf.keras.Model):
    def call(self, y, t):
        return tf.matmul(y, true_A)

f = Model()
true_y = tf.contrib.integrate.odeint(lambda y0, t: f(y0, t), true_y0, t)


def get_batch():
    s = np.random.choice(
        np.arange(data_size - batch_time, dtype=np.int64),
        batch_size, replace=False)
    true_y_np = true_y.numpy()
    batch_y0 = tf.squeeze(tf.convert_to_tensor(true_y_np[s]),1)
    batch_t = t[:batch_time]
    batch_y = tf.squeeze(tf.stack([true_y_np[s + i] for i in range(batch_time)], axis=0),2)
    return batch_y0, batch_t, batch_y

#
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_shape=(2,), activation=tf.nn.relu),
    tf.keras.layers.Dense(2),
])
# model = Model()
model_func = lambda *args, **kwargs: model(*args, **kwargs)
true_y = tf.contrib.integrate.odeint(lambda y0, t: model_func(y0, t), true_y0, t)
print(true_y.shape)
batch_y0, batch_t, batch_y = get_batch()
pred_y = tf.contrib.integrate.odeint(lambda y0, t: model_func(y0, t), batch_y0, batch_t)
print(pred_y.shape)
lr = 1e-3
optimizer = tf.train.AdamOptimizer(lr)
niters = 10
log_freq = 1
for i in range(1, niters + 1):
  batch_y0, batch_t, batch_y = get_batch()
  print(batch_y0.shape, batch_t.shape, batch_y.shape)
  print(len(model.variables))
  with tf.GradientTape() as tape:
    pred_y = odeint(model_func, batch_y0, batch_t)
    loss = tf.reduce_mean(tf.abs(pred_y - batch_y))

  grads = tape.gradient(loss, model.variables)
  grad_vars = zip(grads, model.variables)
  optimizer.apply_gradients(grad_vars)
  if i % log_freq == 0:
    pred_y = odeint(model_func, batch_y0, batch_t)
    loss = tf.reduce_mean(tf.abs(pred_y - batch_y))
    print('Iter {} | Loss {}'.format(i, loss.numpy()))
