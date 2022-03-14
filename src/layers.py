import tensorflow as tf
import numpy as np

class Evolve(tf.keras.layers.Layer):
    def __init__(self, initial_value, trainable=True, name=''):
        super(Evolve, self).__init__()

        self.alpha = tf.Variable(
            initial_value=initial_value,
            trainable=trainable,
            name=name
        )

    def call(self, inputs):
        prev_val = inputs[0]
        new_input = inputs[1]
        return (1 - self.alpha) * prev_val + self.alpha * new_input

class Linear(tf.keras.layers.Layer):
    def __init__(self, initial_weights, bias=False, mask=None, trainable=True, name=''):
        super(Linear, self).__init__()

        self.mask = mask
        self.w = tf.Variable(
            initial_value=initial_weights,
            trainable=trainable,
            name=name+'_w'
        )

        if bias:
            initial_bias = np.zeros((1, initial_weights.shape[1]), dtype=np.float32)
            self.b = tf.Variable(
                initial_value=initial_bias,
                trainable=trainable,
                name=name+'_b'
            )
        else:
            self.b = tf.zeros((), dtype=tf.float32)


    def call(self, inputs):
        w = self.w * self.mask if self.mask is not None else self.w
        return inputs @ w + self.b


class Recurrent(tf.keras.layers.Layer):
    def __init__(self, initial_weights, initial_bias, mask, EI, trainable=True, name=''):
        super(Recurrent, self).__init__()

        self.mask = mask
        self.EI = EI
        self.w = tf.Variable(
            initial_value=initial_weights,
            trainable=trainable,
            name=name+'_w'
        )
        self.b = tf.Variable(
            initial_value=initial_bias,
            trainable=trainable,
            name=name+'_b'
        )

    def call(self, inputs):
        return inputs @ (self.EI @ tf.nn.relu(self.mask * self.w)) + self.b
