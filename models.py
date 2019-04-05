import tensorflow as tf

from .math_utils import pseudo_inverse


class Linear:

    def __call__(self, x):
        return tf.matmul(x, self.w)

    def _solve_linear_system(self, x, y, regularizer):
        return tf.matmul(pseudo_inverse(x, regularizer), y)

    def fit(self, x, y, regularizer):
        '''Fits to a linear model.
        *Args:
          - x: matrix of the inputs to be used, shape=(*, N) where N is the dimension of the inputs
          - y: matrix of wanted outputs, shape=(*, C) where C is the number of outputs
        '''
        assert len(x) == len(y)
        self.w = self._solve_linear_system(x, y, regularizer)


class ELM(Linear):

    def __init__(self, input_size: int, hidden_size: int, nb_classes: int, activation_function: str):
        assert hidden_size > 0 and input_size > 0 and nb_classes > 0
        assert activation_function in ['sigmoid', 'tanh']
        self.insize, self.hidsize, self.nb_classes = input_size, hidden_size, nb_classes
        self.activation_function = activation_function

    def __call__(self, x):
        assert x.shape[1] == self.hidsize
        return tf.matmul(self._calculate_h(x), self.w)

    def _activation(self, x):
        if self.activation_function == 'sigmoid':
            return tf.sigmoid(x)
        elif self.activation_function == 'tanh':
            return tf.tanh(x)

    def _calculate_h(self, x):
        return self.activation(tf.matmul(x, self.v) + self.b)

    def fit(self, x, y, regularizer):
        '''Fits the data to a ELM model
        *Args:
          - x: matrix of the inputs to be used, shape=(*, N) where N is the dimension of the inputs
          - y: matrix of wanted outputs, shape=(*, C) where C is the number of outputs
        '''
        assert len(x) == len(y)
        assert x.shape[1] == self.insize
        assert y.shape[1] == self.nb_classes
        # generates a random matrix V and bias b
        self.v = tf.random.normal(shape=(self.insize, self.hidsize))
        self.b = tf.random.uniform(shape=(self.hidsize,))
        # calculate H
        h = self._calculate_h(x)
        # calculate w
        self.w = self._solve_linear_system(h, y, regularizer)
