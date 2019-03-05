import numpy as np
import tensorflow as tf
from neuralode.odeint import odeint
tf.enable_eager_execution()
tfe = tf.contrib.eager


data_size = 50
batch_time = 10
batch_size = 30

true_y0 = tf.convert_to_tensor([[0.5, 0.3]], dtype=tf.float32)
true_A = tf.convert_to_tensor([[-0.1, -1.], [1., -0.1]],  dtype=tf.float32)
t = tf.linspace(0., 25., data_size)

# TODO with keras layer
# class f(tf.keras.Model):
#     def call(self, y, t):
#         return tf.matmul(y, true_A)

def f(y, t):
    return tf.matmul(y, true_A)

true_y= tf.contrib.integrate.odeint(lambda y0, t: f(y0, t), true_y0, t)

print(true_y.shape)


def get_batch():
    s = np.random.choice(
        np.arange(data_size - batch_time,
                  dtype=np.int64), batch_size,
        replace=False)

    temp_y = true_y.numpy()
    batch_y0 = tf.convert_to_tensor(temp_y[s])
    batch_t = t[:batch_time]
    batch_y = tf.stack([temp_y[s + i] for i in range(batch_time)], axis=0)
    return batch_y0, batch_t, batch_y


class OdeTrained(tf.keras.Model):
    def __init__(self, **kwargs):
        super(OdeTrained, self).__init__(**kwargs)

        self.x = tf.keras.layers.Dense(50, activation='relu',
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.y = tf.keras.layers.Dense(2,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

    def call(self, y, t):
        y = tf.cast(y, tf.float32)
        x = self.x(y)
        y = self.y(x)
        return y


func = OdeTrained()
lr = 1e-3
optimizer = tf.train.AdamOptimizer(lr)
niters = 10
log_freq = 2
for itr in range(1, niters + 1):

    with tf.GradientTape() as tape:
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t)
        loss = tf.reduce_mean(tf.abs(pred_y - batch_y))

    grads = tape.gradient(loss, func.variables)
    grad_vars = zip(grads, func.variables)
    optimizer.apply_gradients(grad_vars)

    if itr % log_freq == 0:
        pred_y = odeint(func, true_y0, t)
        loss = tf.reduce_mean(tf.abs(pred_y - true_y))
        print('Iter {} | Loss {}'.format(itr, loss.numpy()))

