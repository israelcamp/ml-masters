import tensorflow as tf


def return_numpy(func):

    def npvalue(*args, **kwargs):
        return func(*args, **kwargs).numpy()
    return npvalue


@return_numpy
def accuracy(pred: tf.Tensor, true: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(true, 1)), tf.int32))/pred.shape[0]


@return_numpy
def p_mse_loss(pred: tf.Tensor, true: tf.Tensor, p: float = 1.) -> tf.Tensor:
    return tf.reduce_mean(tf.metrics.mean_squared_error(true, pred)**p)
