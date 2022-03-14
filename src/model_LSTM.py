import numpy as np
import copy
from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt
from .layers import Linear, Recurrent, Evolve


class Model():

    def __init__(self, args, learning_type='supervised', n_RFs=2):
        self._args = args
        self._args.n_hidden = 256
        self.model = self.create_network()

    def create_network(self):

        # Set up inputs
        x_input = tf.keras.Input((self._args.n_bottom_up,)) # Bottom-up input
        y_input = tf.keras.Input((self._args.n_top_down_hidden,))
        h_input = tf.keras.Input((1, self._args.n_hidden,)) # Previous activity
        m_input = tf.keras.Input((self._args.n_hidden,)) # Previous activity
        modulator = tf.ones_like(m_input)

        top_down_current = tf.keras.layers.Dense(self._args.n_hidden,
                            trainable=True,
                            name='top_down1')(y_input)

        bottom_up_current = tf.keras.layers.Dense(self._args.n_hidden,
                            trainable=True,
                            name='bottom_up')(x_input)

        rec_current = tf.keras.layers.LSTM(self._args.n_hidden,
                        trainable=True,
                        name='rnn')(h_input)

        h = bottom_up_current + top_down_current + rec_current
        policy = tf.keras.layers.Dense(self._args.n_actions,
                    trainable=True,
                    name='policy')(h)

        policy = tf.nn.softmax(policy, axis=-1)
        critic = tf.keras.layers.Dense(2,
                    trainable=True,
                    name='crtic')(h)

        return tf.keras.models.Model(
            inputs=[x_input, y_input, h_input, m_input],
            outputs=[h, modulator, policy, critic])


    def generate_new_weights(self, new_args, random_seed=1):
        # Set random seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        self._args = new_args
        return

    def initial_activity(self):
        soma_exc = tf.random.uniform((1, self._args.n_exc), 0., self._args.init_exc)
        soma_inh = tf.random.uniform((1, self._args.n_inh), 0., self._args.init_inh)
        soma_init = tf.concat((soma_exc, soma_inh), axis = -1)
        mod_init = tf.random.uniform((1, self._args.n_hidden), 0., self._args.init_mod)
        return soma_init, mod_init

