import numpy as np
import copy
from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt
from layers import Linear, Recurrent, Evolve


class Model():

    def __init__(self, args, learning_type='supervised'):

        self._args = args
        self.learning_type = learning_type
        self.max_weight_value = 2.
        self.top_down_trainable = True if learning_type=='supervised' else False
        self._set_pref_dirs()
        self.model = self.create_network()


    def create_network(self):

        self.EI_list = [1. for i in range(self._args.n_exc)] + \
            [-1. for i in range(self._args.n_inh)]

        self.EI_matrix = tf.linalg.diag(tf.constant(self.EI_list))
        self.mask = tf.ones_like(self.EI_matrix) - tf.linalg.diag(tf.ones(self._args.n_hidden))
        self.EI = tf.cast(self.EI_matrix, tf.float32)
        self.mask = tf.cast(self.mask, tf.float32)

        w_bottom_up = self.initialize_bottom_up_weights()
        w_top_down0, w_top_down1 = self.initialize_top_down_weights()
        w_policy, w_critic = self.initialize_output_weights()
        w_rnn, b_rnn = self.initialize_recurrent_weights()
        w_mod = self.initialize_modulation_weights()
        alpha_soma, alpha_modulator = self.initialize_decay_time_constants()

        noise_rnn_sd = np.sqrt(2/alpha_soma)*self._args.noise_rnn_sd

        x_input = tf.keras.Input((self._args.n_bottom_up,)) # Bottom-up input
        y_input = tf.keras.Input((self._args.n_top_down,)) # Top-down input
        h_input = tf.keras.Input((self._args.n_hidden,)) # Previous activity
        m_input = tf.keras.Input((self._args.n_hidden,)) # Previous activity


        if self.learning_type == 'supervised':

            top_down_hidden = Linear(
                                w_top_down0,
                                trainable=self.top_down_trainable,
                                name='top_down0')(y_input)

            if self._args.nonlinear_top_down:
                top_down_hidden = tf.nn.relu(top_down_hidden)

            top_down_current = Linear(
                                w_top_down1,
                                trainable=False,
                                name='top_down1')(top_down_hidden)

        else:

            print('H', w_top_down1.shape, y_input.shape)
            top_down_current = Linear(
                                w_top_down1,
                                trainable=False,
                                name='top_down1')(y_input)


        bottom_up_current = Linear(
                            w_bottom_up,
                            trainable=False,
                            name='bottom_up')(x_input)


        rec_current = Recurrent(
                        w_rnn,
                        b_rnn,
                        self.mask,
                        self.EI_matrix,
                        trainable=False,
                        name='rnn')(h_input)

        modulation = Linear(
                        w_mod,
                        trainable=False,
                        name='modulation')(h_input)


        modulator = Evolve(alpha_modulator, trainable=False, name='alpha_modulator')((m_input, modulation))
        effective_mod = 1 / (1 + tf.nn.relu(modulator))

        noise = tf.random.normal(tf.shape(h_input), mean=0., stddev=noise_rnn_sd)
        soma_input = bottom_up_current + top_down_current + rec_current * effective_mod + noise
        h = Evolve(alpha_soma, trainable=False, name='alpha_soma')((h_input, soma_input))
        h = tf.nn.relu(h)
        h_exc = h[..., :self._args.n_exc]
        if self.learning_type == 'RL':
            h_exc = tf.clip_by_value(h_exc, 0., self._args.max_h_for_output)

        policy = Linear(
                    w_policy,
                    trainable=True,
                    bias=True,
                    name='policy')(h_exc)

        if self.learning_type == 'RL':

            policy = tf.nn.softmax(policy, axis=-1)

            critic = Linear(
                        w_critic,
                        trainable=True,
                        bias=True,
                        name='crtic')(h_exc)

            return tf.keras.models.Model(
                inputs=[x_input, y_input, h_input, m_input],
                outputs=[h, modulator, policy, critic])

        elif self.learning_type == 'supervised':

            return tf.keras.models.Model(
                inputs=[x_input, y_input, h_input, m_input],
                outputs=[h, modulator, policy])


    def generate_new_weights(self, new_args):

        self._args = new_args
        w_bottom_up = self.initialize_bottom_up_weights()

        w_top_down0, w_top_down1 = self.initialize_top_down_weights()
        w_policy, w_critic = self.initialize_output_weights()
        w_rnn, b_rnn = self.initialize_recurrent_weights()
        w_mod = self.initialize_modulation_weights()
        alpha_soma, alpha_modulator = self.initialize_decay_time_constants()
        b_pol = np.float32([1., 0., 0.])[np.newaxis, :]

        var_reset_dict = {'modulation_w:0': w_mod, 'bottom_up_w:0': w_bottom_up, \
            'rnn_w:0': w_rnn, 'rnn_b:0': b_rnn, 'top_down0_w:0': w_top_down0, \
            'top_down1_w:0': w_top_down1, 'policy_w:0': w_policy, 'policy_b:0': b_pol, \
            'critic_w:0': w_critic, 'critic_b:0': None, \
            'alpha_modulator:0': alpha_modulator, 'alpha_soma:0': alpha_soma}

        for var in self.model.non_trainable_variables + self.model.trainable_variables:
            for key, reset_val in var_reset_dict.items():
                if var.name == key:
                    print(f'Resetting {var.name}')
                    if reset_val is not None:
                        var.assign(reset_val)
                    else:
                        var.assign(tf.zeros_like(var))


    def intial_activity(self):

        soma_exc = tf.random.uniform((1, self._args.n_exc), 0., self._args.init_exc)
        soma_inh = tf.random.uniform((1, self._args.n_inh), 0., self._args.init_inh)
        soma_init = tf.concat((soma_exc, soma_inh), axis = -1)
        mod_init = tf.random.uniform((1, self._args.n_hidden), 0., self._args.init_mod)

        return soma_init, mod_init


    def _set_pref_dirs(self):

        '''Motion direcion input neurons will have prefered direction, but
        fixation neurons will not. Fixation neurons will be at the end of
        the array'''
        input_phase = np.linspace(0, 2*np.pi, self._args.n_motion_tuned)
        exc_phase = np.linspace(0, 2*np.pi, self._args.n_exc)
        inh_phase = np.linspace(0, 2*np.pi, self._args.n_inh)
        rnn_phase = np.concatenate((exc_phase, inh_phase), axis=-1)

        self._inp_rnn_phase = np.cos(input_phase[:, np.newaxis] - rnn_phase[np.newaxis, :])
        N = self._args.n_bottom_up - self._args.n_motion_tuned
        self._inp_rnn_phase = np.vstack((self._inp_rnn_phase, np.zeros((N, self._args.n_hidden), dtype = np.float32)))
        self._rnn_rnn_phase = np.cos(rnn_phase[:, np.newaxis] - rnn_phase[np.newaxis, :])


    def initialize_decay_time_constants(self):

        alpha_soma = np.clip(self._args.dt / self._args.tc_soma, 0., 1.)
        alpha_modulator = np.clip(self._args.dt/self._args.tc_modulator, 0, 1.)
        return np.float32(alpha_soma), np.float32(alpha_modulator)


    def initialize_top_down_weights(self):


        initializer = tf.keras.initializers.GlorotNormal()
        w0 = initializer(shape=(self._args.n_top_down, self._args.n_top_down_hidden))


        We = np.random.gamma(
                        self._args.td_E_kappa,
                        1.,
                        size = (self._args.n_top_down_hidden, self._args.n_exc))
        Wi = np.random.gamma(
                        self._args.td_I_kappa,
                        1.,
                        size = (self._args.n_top_down_hidden, self._args.n_inh))


        w1 = self._args.td_weight * np.concatenate((We, Wi), axis=1)
        w1 = np.clip(w1, 0., self.max_weight_value)

        return tf.cast(w0, tf.float32), tf.cast(w1, tf.float32)

    def initialize_output_weights(self):

        initializer = tf.keras.initializers.GlorotNormal()
        #w_policy = initializer(shape=(self._args.n_hidden, self._args.n_actions))
        #w_critic = initializer(shape=(self._args.n_hidden, 2))
        w_policy = initializer(shape=(self._args.n_exc, self._args.n_actions))
        w_critic = initializer(shape=(self._args.n_exc, 2))

        return tf.cast(w_policy, tf.float32), tf.cast(w_critic, tf.float32)


    def initialize_bottom_up_weights(self):

        '''Input neurons will project unfiformly to the half the EXC neurons;
        input strength determined by Von Mises distribution.
        Inputs only project to every second neuron'''

        We = np.random.gamma(
                        self._args.inp_E_kappa,
                        1.,
                        size = (self._args.n_bottom_up, self._args.n_exc))
        Wi = np.random.gamma(
                        self._args.inp_I_kappa,
                        1.,
                        size = (self._args.n_bottom_up, self._args.n_inh))

        N = self._args.n_motion_tuned

        We[:N, :] *= von_mises(
                    self._inp_rnn_phase[:N, :self._args.n_exc],
                    self._args.inp_E_topo)
        Wi[:N, :] *= von_mises(
                    self._inp_rnn_phase[:N, self._args.n_exc:],
                    self._args.inp_I_topo)



        W = np.concatenate((We, Wi), axis=1)
        W *= self._args.input_weight
        W = np.clip(W, 0., self.max_weight_value)

        return np.float32(W)

    def initialize_modulation_weights(self):


        # THIS HAS RECENTLY CHANGED!
        """
        beta_EE = self._args.mod_EE_weight
        beta_EI = self._args.mod_EI_weight
        beta_IE = self._args.mod_IE_weight
        beta_II = self._args.mod_II_weight
        """
        beta_EE = self._args.mod_EE_weight / self._args.n_exc
        beta_EI = self._args.mod_EI_weight / self._args.n_inh
        beta_IE = self._args.mod_IE_weight / self._args.n_exc
        beta_II = self._args.mod_II_weight / self._args.n_inh

        Wee = von_mises(
                    self._rnn_rnn_phase[:self._args.n_exc, :self._args.n_exc],
                    self._args.EE_topo_mod)
        Wei = von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, :self._args.n_exc],
                    self._args.EI_topo_mod)
        Wie = von_mises(
                    self._rnn_rnn_phase[:self._args.n_exc, self._args.n_exc:],
                    self._args.IE_topo_mod)
        Wii = von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, self._args.n_exc:],
                    self._args.II_topo_mod)

        We = np.hstack((beta_EE * Wee, beta_IE * Wie))
        Wi = np.hstack((beta_EI * Wei, beta_II * Wii))
        W = np.vstack((We, Wi))

        return np.float32(W)


    def initialize_recurrent_weights(self):


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
                    self._args.EE_topo)
        Wei *= von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, :self._args.n_exc],
                    self._args.EI_topo)
        Wie *= von_mises(
                    self._rnn_rnn_phase[:self._args.n_exc, self._args.n_exc:],
                    self._args.IE_topo)
        Wii *= von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, self._args.n_exc:],
                    self._args.II_topo)


        We = np.hstack((Wee, Wie))
        Wi = np.hstack((Wei, Wii))
        w_rnn = np.vstack((We, Wi))
        w_rnn *= self._args.rnn_weight

        b_rnn = np.zeros((self._args.n_hidden), dtype = np.float32)

        for i in range(self._args.n_hidden):
            w_rnn[i, i] = 0.

        w_rnn = np.clip(w_rnn, 0., self.max_weight_value)

        return np.float32(w_rnn), np.float32(b_rnn)



def von_mises(phase, kappa):

    x = np.exp(kappa * phase) / np.exp(kappa)
    return x / np.mean(x)
