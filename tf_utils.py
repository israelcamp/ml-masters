import tensorflow as tf


def make_label_array_from_dict(tensors_dict):
    return tf.concat([
        tf.constant(len(xi)*[i]) for i, xi in tensors_dict.items()
    ], axis=0)


def concat_tensors_from_dict(tensors_dict, axis=0):
    return tf.concat([
        xi for xi in tensors_dict.values()
    ], axis=axis)


def separate_by_class(X: tf.Tensor, Y: tf.Tensor, classes: int, ratio: float = 0.8) -> dict:
    assert 0 < ratio < 1
    train_tensors, valid_tensors = {}, {}
    for i in range(classes):
        ci = tf.equal(Y, tf.constant(len(Y) * [i]))
        xi = X[ci]
        sz = round(len(xi) * ratio)
        train_tensors[i] = xi[:sz]
        valid_tensors[i] = xi[sz:]

    x_train, y_train = concat_tensors_from_dict(
        train_tensors), make_label_array_from_dict(train_tensors)
    x_valid, y_valid = concat_tensors_from_dict(
        valid_tensors), make_label_array_from_dict(valid_tensors)
    return x_train, y_train, x_valid, y_valid


def append_const_column(x: tf.Tensor, alpha: float, mode: str = 'first') -> tf.Tensor:
    assert mode in ['first', 'last']

    def op():
        if mode == 'first':
            return alpha * tf.ones(shape=(x.shape[0], 1)), x
        elif mode == 'last':
            return x, alpha * tf.ones(shape=(x.shape[0], 1))
    return tf.concat(op(), axis=1)


def shuffle(*tensors, seed=0):
    return [
        tf.random.shuffle(t, seed) for t in tensors
    ]
