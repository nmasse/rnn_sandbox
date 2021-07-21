import numpy as np
import copy
from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense


class Model():

    def __init__(self, args):

        self._args = args
        self._args.n_hidden = self._args.n_exc + self._args.n_inh

        self._set_pref_dirs()
        self.model = self.create_network()


    def create_network(self):

        self.EI_list = [1. for i in range(self._args.n_exc)] + \
            [-1. for i in range(self._args.n_inh)]

        self.EI_matrix = tf.linalg.diag(tf.constant(self.EI_list))
        self.mask = tf.ones_like(self.EI_matrix) - tf.linalg.diag(tf.ones(self._args.n_hidden))
        self.EI = tf.cast(self.EI_matrix, tf.float32)
        self.mask = tf.cast(self.mask, tf.float32)

        w_in = self._initialize_input_weights()
        w_rnn, b_rnn = self._initialize_recurrent_weights()

        alpha_soma = self._args.dt / self._args.tc_soma
        alpha_modulator = self._args.dt / self._args.tc_modulator
        noise_rnn_sd = np.sqrt(2/alpha_soma)*self._args.noise_rnn_sd

        x_input = tf.keras.Input((self._args.n_input,)) # Stimulus input
        s_input = tf.keras.Input((self._args.n_hidden,)) # Soma
        m_input = tf.keras.Input((1,)) # Modulator

        h = tf.nn.relu(s_input) # Spiking activity
        soma = (1 - alpha_soma) * s_input
        modulator = (1 - alpha_modulator) * m_input

        input_current = Linear(w_in, name='input')(x_input)
        #input_current = Dense(w_in, name='input')(x_input)
        rec_current = Recurrent(w_rnn, b_rnn, self.mask, self.EI_matrix, name='rnn')(h)
        modulation = tf.reduce_sum(h[:, :self._args.n_exc], axis=-1, keepdims=True)
        #modulation = tf.reduce_sum(h, axis=-1, keepdims=True)

        modulator += alpha_modulator * modulation
        effective_mod = 1 / (1 + self._args.mod_beta * tf.nn.relu(modulator))

        noise = tf.random.normal(tf.shape(h), mean=0., stddev=noise_rnn_sd)
        #m = effective_mod ** tf.nn.relu(tf.constant(self.EI_list))
        #print('M',m.shape)
        #rec_current *= m
        #print('X', rec_current.shape, effective_mod.shape)
        rec_current *= effective_mod

        soma += alpha_soma * (input_current + rec_current + noise)
        # TODO: Think about bias term and modulation!!!!!!!!!!!!!!!!!
        #soma = tf.nn.relu(soma)
        return tf.keras.models.Model(
            inputs=[x_input, s_input, m_input],
            outputs=[soma, modulator])


    def generate_new_weights(self):

        w_in = self._initialize_input_weights()
        w_rnn, b_rnn = self._initialize_recurrent_weights()
        for v in self.model.trainable_variables:
            print(v.name)
        1/0




    def intial_activity(self, batch_size):

        soma_exc = tf.random.uniform((batch_size, self._args.n_exc), 0., self._args.init_exc)
        soma_inh = tf.random.uniform((batch_size, self._args.n_inh), 0., self._args.init_inh)
        soma_init = tf.concat((soma_exc, soma_inh), axis = -1)
        mod_init = tf.random.uniform((batch_size, 1), 0., self._args.init_mod)

        return soma_init, mod_init


    def _set_pref_dirs(self):

        '''Motion direcion input neurons will have prefered direction, but
        fixation neurons will not. Fixation neurons will be at the end of
        the array'''

        input_phase = np.linspace(0, 2*np.pi, self._args.n_input-self._args.n_fix-self._args.n_rule)
        exc_phase = np.linspace(0, 2*np.pi, self._args.n_exc)
        inh_phase = np.linspace(0, 2*np.pi, self._args.n_inh)
        rnn_phase = np.concatenate((exc_phase, inh_phase), axis=-1)

        self._inp_rnn_phase = np.cos(input_phase[:, np.newaxis] - rnn_phase[np.newaxis, :])
        self._inp_rnn_phase = np.concatenate((self._inp_rnn_phase , np.zeros((self._args.n_fix+self._args.n_rule, self._args.n_hidden))))
        self._rnn_rnn_phase = np.cos(rnn_phase[:, np.newaxis] - rnn_phase[np.newaxis, :])


    def _initialize_input_weights(self):

        '''Input neurons will project unfiformly to the half the EXC neurons;
        input strength determined by Von Mises distribution.
        Inputs only project to every second neuron'''

        W = np.zeros((self._args.n_input, self._args.n_hidden), dtype = np.float32)
        N = self._args.n_input-self._args.n_fix-self._args.n_rule
        W[:N, :self._args.n_exc] = von_mises(
                                        self._inp_rnn_phase[:N, :self._args.n_exc],
                                        self._args.inp_E_topo,
                                        self._args.inp_E_weight)
        #W[:N, :self._args.n_exc] -= np.mean(W[:N, :self._args.n_exc])
        W[:N, self._args.n_exc:] = von_mises(
                                        self._inp_rnn_phase[:N, self._args.n_exc:],
                                        self._args.inp_I_topo,
                                        self._args.inp_I_weight)
        #W[:N, self._args.n_exc:] -= np.mean(W[:N, self._args.n_exc:])

        #W[-self._args.n_rule:, :] += np.random.normal(0, 0.1, size=W[-self._args.n_rule:, :].shape)
        #W[N:, :] = np.random.normal(0, 0.1, size=W[N:, :].shape)
        W[N:, :self._args.n_exc] = np.random.uniform(0., 0.1, W[N:, :self._args.n_exc].shape)
        W[N:, self._args.n_exc:] = np.random.uniform(0., 0.2, W[N:, self._args.n_exc:].shape)
        #W[:N, :] += np.random.normal(0, 0.1, size=W[:N, :].shape)
        #W += np.random.normal(0, 0.1, size=W.shape)

        """
        plt.imshow(W,aspect='auto')
        plt.colorbar()
        plt.show()
        """

        return W


    def _initialize_recurrent_weights(self):


        Wee = np.random.gamma(self._args.EE_kappa, 1., size = (self._args.n_exc, self._args.n_exc))
        Wie = np.random.gamma(self._args.IE_kappa, 1., size = (self._args.n_exc, self._args.n_inh))
        Wei = np.random.gamma(self._args.EI_kappa, 1., size = (self._args.n_inh, self._args.n_exc))
        Wii = np.random.gamma(self._args.II_kappa, 1., size = (self._args.n_inh, self._args.n_inh))


        # Controlling the strength of the reciprocal connection
        Wie_temp = copy.copy(Wie)
        Wie += (self._args.alpha_EI * np.transpose(Wei))
        Wei += (self._args.alpha_EI * np.transpose(Wie_temp))
        Wee += (self._args.alpha_EE * np.transpose(Wee))
        Wii += (self._args.alpha_II * np.transpose(Wii))
        Wee /= (1 + self._args.alpha_EE)
        Wie /= (1 + self._args.alpha_EI)
        Wei /= (1 + self._args.alpha_EI)
        Wii /= (1 + self._args.alpha_II)

        # Applying the topological modifier
        Wee *= von_mises(
                    self._rnn_rnn_phase[:self._args.n_exc, :self._args.n_exc],
                    self._args.EE_topo,
                    self._args.rnn_mult_factor)
        Wei *= von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, :self._args.n_exc],
                    self._args.EI_topo,
                    self._args.rnn_mult_factor)
        Wie *= von_mises(
                    self._rnn_rnn_phase[:self._args.n_exc, self._args.n_exc:],
                    self._args.IE_topo,
                    self._args.rnn_mult_factor)
        Wii *= von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, self._args.n_exc:],
                    self._args.II_topo,
                    self._args.rnn_mult_factor)


        We = np.hstack((Wee, Wie))
        Wi = np.hstack((Wei, Wii))
        w_rnn = np.vstack((We, Wi))

        b_rnn = np.zeros((self._args.n_hidden), dtype = np.float32)
        b_rnn[:self._args.n_exc] = np.random.normal(
            self._args.E_bias_mean,
            self._args.E_bias_std,
            size = (self._args.n_exc))

        b_rnn[self._args.n_exc:] = np.random.normal(
            self._args.I_bias_mean,
            self._args.I_bias_std,
            size = (self._args.n_inh))

        for i in range(self._args.n_hidden):
            w_rnn[i, i] = 0.

        w_rnn = np.clip(w_rnn, 0., 1.)

        return np.float32(w_rnn), np.float32(b_rnn)



class LSTM(tf.keras.layers.Layer):

    def __init__(
        self,
        n_input,
        n_hidden,
        name=None):

        super().__init__(name=name)

        init = tf.keras.initializers.Orthogonal(gain=1.0)
        init_z = tf.keras.initializers.Zeros()

        self.wf = tf.Variable(initial_value = init([n_input, n_hidden]))
        self.wi = tf.Variable(initial_value = init([n_input, n_hidden]))
        self.wo = tf.Variable(initial_value = init([n_input, n_hidden]))
        self.wc = tf.Variable(initial_value = init([n_input, n_hidden]))
        self.uf = tf.Variable(initial_value = init([n_hidden, n_hidden]))
        self.ui = tf.Variable(initial_value = init([n_hidden, n_hidden]))
        self.uo = tf.Variable(initial_value = init([n_hidden, n_hidden]))
        self.uc = tf.Variable(initial_value = init([n_hidden, n_hidden]))
        self.bf = tf.Variable(initial_value = init_z([1, n_hidden]))
        self.bi = tf.Variable(initial_value = init_z([1, n_hidden]))
        self.bo = tf.Variable(initial_value = init_z([1, n_hidden]))
        self.bc = tf.Variable(initial_value = init_z([1, n_hidden]))

    def call(self, x, h, c_old):

        f = tf.nn.sigmoid(x @ self.wf + h @ self.uf + self.bf)
        i = tf.nn.sigmoid(x @ self.wi + h @ self.ui + self.bi)
        o = tf.nn.sigmoid(x @ self.wo + h @ self.uo + self.bo)
        c = tf.nn.tanh(x @ self.wc + h @ self.uc + self.bc)

        c_new = f * c_old + i * c
        h = o * tf.nn.tanh(c_new)

        return h, c_new


def von_mises(phase, kappa, alpha):

    x = np.exp(kappa * phase) / np.exp(kappa)
    x /= np.mean(x)
    return alpha * x

class Linear(tf.keras.layers.Layer):
    def __init__(self, initial_weights, mask=None, name=''):
        super(Linear, self).__init__()

        self.mask = mask
        self.w = tf.Variable(
            initial_value=initial_weights,
            trainable=True,
            name=name+'_w'
        )

    def call(self, inputs):
        w = self.w * self.mask if self.mask is not None else self.w
        return inputs @ w


class Recurrent(tf.keras.layers.Layer):
    def __init__(self, initial_weights, initial_bias, mask, EI, name=''):
        super(Recurrent, self).__init__()

        self.mask = mask
        self.EI = EI
        self.w = tf.Variable(
            initial_value=initial_weights,
            trainable=True,
            name=name+'_w'
        )
        self.b = tf.Variable(
            initial_value=initial_bias,
            trainable=True,
            name=name+'_b'
        )

    def call(self, inputs):
        return inputs @ (self.EI @ tf.nn.relu(self.mask * self.w)) + self.b
