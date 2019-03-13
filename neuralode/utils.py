import tensorflow as tf

def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        tf.reshape(p, [-1]) if p is not None else tf.reshape(tf.zeros_like(q), [-1])
        for p, q in zip(sequence, like_sequence)
    ]
    return tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])


def _flatten(sequence):
    flat = [tf.reshape(p, [-1]) for p in sequence]
    return tf.concat(flat, 0) if len(flat) > 0 else tf.convert_to_tensor([])


def euler(func, y0, t, rtol=1e-6, atol=1e-12):
  yt = [y0]
  for tprev, tnext in zip(t[:-1], t[1:]):
    yt.append(yt[-1] + (tnext - tprev) * func(yt[-1], tnext))
  return tf.stack(yt)

