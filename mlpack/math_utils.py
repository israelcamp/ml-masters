import tensorflow as tf


def pseudo_inverse(A, c=0):
    return tf.matmul(tf.linalg.inv(tf.add(tf.matmul(A, A, transpose_a=True), c * tf.eye(A.shape[1]))), A, transpose_b=True)
