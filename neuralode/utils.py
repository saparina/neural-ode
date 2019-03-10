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

