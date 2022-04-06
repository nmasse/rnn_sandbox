import numpy as np
import copy
from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt
from .layers import Linear, Recurrent, Evolve
from . import util

class Model():

    def __init__(self, args, learning_type='supervised', n_RFs=2, **kwargs):
        self._args = args
        self.n_RFs = n_RFs
        self.learning_type = learning_type
        self.max_weight_value = 2.
        self.top_down_trainable = (learning_type == "supervised") and self._args.top_down_trainable
        self.tau_fast = 200
        self.tau_slow = 1500
        self._set_pref_dirs()
        self.model = self.create_network()

    def create_network(self):

        self.EI_list = [1. for i in range(self._args.n_exc)] + \
            [-1. for i in range(self._args.n_inh)]

        self.EI_matrix = tf.linalg.diag(tf.constant(self.EI_list))
        self.mask = tf.ones_like(self.EI_matrix) - tf.linalg.diag(tf.ones(self._args.n_hidden))
        self.EI = tf.cast(self.EI_matrix, tf.float32)
        self.mask = tf.cast(self.mask, tf.float32)

        # Initialize STSP values
        self.alpha_stf = np.ones((1, self._args.n_hidden), dtype=np.float32)
        self.alpha_std = np.ones((1, self._args.n_hidden), dtype=np.float32)
        self.U = np.ones((1, self._args.n_hidden), dtype=np.float32)
        self.syn_x_init = np.ones((1, self._args.n_hidden), dtype=np.float32)
        self.syn_u_init = 0.3 * np.ones((1, self._args.n_hidden), dtype=np.float32)
        self.dynamic_synapse = np.zeros((1, self._args.n_hidden), dtype=np.float32)

        randperm_exc = np.random.permutation(self._args.n_exc)
        randperm_inh = np.random.permutation(self._args.n_inh)
        randperm_inh += self._args.n_exc

        # Use different approach for fac/dep: instead of doing every
        # other, which overlaps w/ some other approaches we take,
        # separately permute E and I orderings, then take first
        # half of each random ordering and make facilitating, 
        # second half depressing
        for indices in [randperm_exc, randperm_inh]:
            for i, ind in enumerate(indices):
                if i < len(indices) // 2:
                    self.alpha_stf[0,ind] = self._args.dt/self.tau_slow
                    self.alpha_std[0,ind] = self._args.dt/self.tau_fast
                    self.U[0,ind] = 0.15
                    self.syn_u_init[:, ind] = self.U[0,ind]
                    self.dynamic_synapse[0,ind] = 1
                else:
                    self.alpha_stf[0,ind] = self._args.dt/self.tau_fast
                    self.alpha_std[0,ind] = self._args.dt/self.tau_slow
                    self.U[0,ind] = 0.45
                    self.syn_u_init[:, ind] = self.U[0,ind]
                    self.dynamic_synapse[0,ind] = 1


        '''
        for i in range(self._args.n_hidden):
            #if i % 2 == 0:
            if randperm[i] > self._args.n_hidden
                self.alpha_stf[0,i] = self._args.dt/self.tau_slow
                self.alpha_std[0,i] = self._args.dt/self.tau_fast
                self.U[0,i] = 0.15
                self.syn_u_init[:, i] = self.U[0,i]
                self.dynamic_synapse[0,i] = 1

            else:
                self.alpha_stf[0,i] = self._args.dt/self.tau_fast
                self.alpha_std[0,i] = self._args.dt/self.tau_slow
                self.U[0,i] = 0.45
                self.syn_u_init[:, i] = self.U[0,i]
                self.dynamic_synapse[0,i] = 1
        '''

        # Make into constants
        self.alpha_stf = tf.constant(self.alpha_stf)
        self.alpha_std = tf.constant(self.alpha_std)
        self.U = tf.constant(self.U)
        self.syn_x_init = tf.constant(self.syn_x_init)
        self.syn_u_init = tf.constant(self.syn_u_init)
        self.dynamic_synapse = tf.constant(self.dynamic_synapse)


        w_bottom_up = self.initialize_bottom_up_weights()
        w_policy, w_critic = self.initialize_output_weights()
        w_top_down0, w_top_down1 = self.initialize_top_down_weights(w_policy)
        w_rnn, b_rnn = self.initialize_recurrent_weights()
        w_mod = self.initialize_modulation_weights()
        alpha_soma, alpha_modulator = self.initialize_decay_time_constants()


        noise_rnn_sd = np.sqrt(2/alpha_soma)*self._args.noise_rnn_sd

        x_input = tf.keras.Input((self._args.n_bottom_up,)) # Bottom-up input
        if self.learning_type == 'supervised':
            y_input = tf.keras.Input((self._args.n_top_down,)) # Top-down input
        else:
            y_input = tf.keras.Input((self._args.n_top_down_hidden,))
        h_input = tf.keras.Input((self._args.n_hidden,)) # Previous activity
        m_input = tf.keras.Input((self._args.n_hidden,)) # Previous activity
        syn_x_input = tf.keras.Input((self._args.n_hidden,)) # Previous synaptic state
        syn_u_input = tf.keras.Input((self._args.n_hidden,))


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

            top_down_current = Linear(
                                w_top_down1,
                                trainable=False,
                                name='top_down1')(y_input)


        bottom_up_current = Linear(
                            w_bottom_up,
                            trainable=False,
                            name='bottom_up')(x_input)

        # Add in STSP effect
        syn_x = syn_x_input + (self.alpha_std*(1-syn_x_input) - 
            (self._args.dt / 1000) * syn_u_input * syn_x_input * h_input) \
            * self.dynamic_synapse
        syn_u = syn_u_input + (self.alpha_stf*(self.U-syn_u_input) + 
            (self._args.dt / 1000) * self.U * (1-syn_u_input) * h_input) \
            * self.dynamic_synapse
        syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
        syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
        h_post = syn_u * syn_x * h_input


        rec_current = Recurrent(
                        w_rnn,
                        b_rnn,
                        self.mask,
                        self.EI_matrix,
                        trainable=False,
                        name='rnn')(h_post)

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
        h_out = h[..., :self._args.n_exc] if self._args.restrict_output_to_exc else h
        #if self.learning_type == 'RL':
        h_out = tf.clip_by_value(h_out, 0., self._args.max_h_for_output)

        policy = Linear(
                    w_policy,
                    trainable=True,
                    bias=True,
                    name='policy')(h_out)

        if self.learning_type == 'RL':

            policy = tf.nn.softmax(policy, axis=-1)

            critic = Linear(
                        w_critic,
                        trainable=True,
                        bias=True,
                        name='crtic')(h_out)

            return tf.keras.models.Model(
                inputs=[x_input, y_input, h_input, m_input, syn_x_input, syn_u_input],
                outputs=[h, modulator, policy, critic, syn_x, syn_u])

        elif self.learning_type == 'supervised':

            return tf.keras.models.Model(
                inputs=[x_input, y_input, h_input, m_input, syn_x_input, syn_u_input],
                outputs=[h, modulator, policy, syn_x, syn_u])


    def generate_new_weights(self, new_args, random_seed=1):

        # Set random seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        self._args = new_args
        w_bottom_up = self.initialize_bottom_up_weights()

        w_policy, w_critic = self.initialize_output_weights()
        w_top_down0, w_top_down1 = self.initialize_top_down_weights(w_policy)
        
        w_rnn, b_rnn = self.initialize_recurrent_weights()
        w_mod = self.initialize_modulation_weights()
        alpha_soma, alpha_modulator = self.initialize_decay_time_constants()
        b_pol = np.float32(np.zeros(self._args.n_actions))
        b_pol[0] = 1.
        b_pol = b_pol[np.newaxis,:]

        var_reset_dict = {'modulation_w:0'   : w_mod, 
                          'bottom_up_w:0'    : w_bottom_up,
                          'rnn_w:0'          : w_rnn, 
                          'rnn_b:0'          : b_rnn, 
                          'top_down0_w:0'    : w_top_down0,
                          'top_down1_w:0'    : w_top_down1, 
                          'policy_w:0'       : w_policy, 
                          'policy_b:0'       : b_pol, 
                          'critic_w:0'       : w_critic, 
                          'critic_b:0'       : None, 
                          'alpha_modulator:0': alpha_modulator, 
                          'alpha_soma:0'     : alpha_soma}

        all_v = self.model.non_trainable_variables + \
            self.model.trainable_variables
        for var in all_v:
            for key, reset_val in var_reset_dict.items():
                if var.name == key:
                    print(f'Resetting {var.name}')
                    if reset_val is not None:
                        var.assign(reset_val)
                    else:
                        var.assign(tf.zeros_like(var))


    def initial_activity(self):

        soma_exc = tf.random.uniform((1, self._args.n_exc), 0., self._args.init_exc)
        soma_inh = tf.random.uniform((1, self._args.n_inh), 0., self._args.init_inh)
        soma_init = tf.concat((soma_exc, soma_inh), axis = -1)
        mod_init = tf.random.uniform((1, self._args.n_hidden), 0., self._args.init_mod)

        # Initial synaptic values
        syn_x_init = self.syn_x_init
        syn_u_init = self.syn_u_init

        return soma_init, mod_init, syn_x_init, syn_u_init


    def _set_pref_dirs(self):

        '''Motion direcion input neurons will have prefered direction, but
        fixation neurons will not. Fixation neurons will be at the end of
        the array'''
        N = self._args.n_bottom_up - self._args.n_motion_tuned

        rnn_phase = np.linspace(0, 2*np.pi, self._args.n_hidden)
        rnn_phase0 = rnn_phase[np.random.permutation(self._args.n_hidden)]
        rnn_phase1 = rnn_phase[np.random.permutation(self._args.n_hidden)]

        # Trying to add spatial topology here
        #if self._args.n_cue_tuned == 2:
        if self.n_RFs == 2 and self._args.bottom_up_topology:
            # Two RFs
            motion_phase = np.linspace(0, 2*np.pi, self._args.n_motion_tuned//2)
            motion_phase = np.concatenate((motion_phase, motion_phase), axis=-1)
            RF_phase = self._args.n_motion_tuned//2 * [0] + self._args.n_motion_tuned//2 * [np.pi]
            RF_phase = np.array(RF_phase)
            
            RF_rnn = np.cos(RF_phase[:, np.newaxis] - rnn_phase1[np.newaxis, :])
            RF_rnn = np.vstack((RF_rnn, np.zeros((N, self._args.n_hidden))))

        elif self._args.n_cue_tuned<=1 or not self._args.bottom_up_topology:
            # One RF
            motion_phase = np.linspace(0, 2*np.pi, self._args.n_motion_tuned)
            RF_rnn =  np.zeros((self._args.n_bottom_up, self._args.n_hidden))
        elif self._args.n_cue_tuned > 2:
            assert False, "Not sure how to handle 3 or more cue tuned neurons"

        print(f"Bottom up and motion tuned {self._args.n_bottom_up} , {self._args.n_motion_tuned}")

        td_phase = np.linspace(0, 2*np.pi, self._args.n_top_down_hidden)
        td_phase0 = td_phase[np.random.permutation(self._args.n_top_down_hidden)]
        td_phase1 = td_phase[np.random.permutation(self._args.n_top_down_hidden)]

        motion_rnn = np.cos(motion_phase[:, np.newaxis] - rnn_phase0[np.newaxis, :])
        motion_rnn = np.vstack((motion_rnn, np.zeros((N, self._args.n_hidden))))

        self._inp_rnn_phase = 0.5*motion_rnn + 0.5*RF_rnn

        self._rnn_rnn_phase = 0.5*np.cos(rnn_phase0[:, np.newaxis] - rnn_phase0[np.newaxis, :]) \
            + 0.5*np.cos(rnn_phase1[:, np.newaxis] - rnn_phase1[np.newaxis, :])
        self._td_rnn_phase = 0.5*np.cos(td_phase0[:, np.newaxis] - rnn_phase0[np.newaxis, :]) \
            + 0.5*np.cos(td_phase1[:, np.newaxis] - rnn_phase1[np.newaxis, :])



    def initialize_decay_time_constants(self):

        alpha_soma = np.clip(self._args.dt / self._args.tc_soma, 0., 1.)
        alpha_modulator = np.clip(self._args.dt/self._args.tc_modulator, 0, 1.)
        return np.float32(alpha_soma), np.float32(alpha_modulator)


    def initialize_top_down_weights(self, w_policy):

        initializer = tf.keras.initializers.GlorotNormal()
        if self._args.top_down_overlapping:
            w0_temp = initializer(shape=(1, self._args.n_top_down_hidden))
            w0 = tf.tile(w0_temp, [self._args.n_top_down, 1])
        else:
            w0 = initializer(shape=(self._args.n_top_down, self._args.n_top_down_hidden))
        

        We = np.random.gamma(
                        self._args.td_E_kappa,
                        1.,
                        size = (self._args.n_top_down_hidden, self._args.n_exc))
        Wi = np.random.gamma(
                        self._args.td_I_kappa,
                        1.,
                        size = (self._args.n_top_down_hidden, self._args.n_inh))

        We *= von_mises(
                self._td_rnn_phase[:, :self._args.n_exc],
                self._args.td_E_topo)
        Wi *= von_mises(
                self._td_rnn_phase[:, self._args.n_exc:],
                self._args.td_I_topo)

        w1 = self._args.td_weight * np.concatenate((We, Wi), axis=1)
        w1 = np.clip(w1, 0., self.max_weight_value)
        w1[:, 1::2] = 0.

        return tf.cast(w0, tf.float32), tf.cast(w1, tf.float32)

    def initialize_output_weights(self):

        initializer = tf.keras.initializers.GlorotNormal()
        if self._args.restrict_output_to_exc:
            w_policy = initializer(shape=(self._args.n_exc, self._args.n_actions))
            w_critic = initializer(shape=(self._args.n_exc, 2))
        else:
            w_policy = initializer(shape=(self._args.n_hidden, self._args.n_actions))
            w_critic = initializer(shape=(self._args.n_hidden, 2))

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

        We[:, :] *= von_mises(
                    self._inp_rnn_phase[:, :self._args.n_exc],
                    self._args.inp_E_topo)
        Wi[:, :] *= von_mises(
                    self._inp_rnn_phase[:, self._args.n_exc:],
                    self._args.inp_I_topo)


        W = np.concatenate((We, Wi), axis=1)
        W *= self._args.input_weight
        W = np.clip(W, 0., self.max_weight_value)

        W[:, ::2] = 0.


        return np.float32(W)

    def initialize_modulation_weights(self):

        beta_EE = self._args.mod_EE_weight / self._args.n_hidden
        beta_EI = self._args.mod_EI_weight / self._args.n_hidden
        beta_IE = self._args.mod_IE_weight / self._args.n_hidden
        beta_II = self._args.mod_II_weight / self._args.n_hidden

        Wee = np.ones((self._args.n_exc, self._args.n_exc))
        Wei = np.ones((self._args.n_inh, self._args.n_exc))
        Wie = np.ones((self._args.n_exc, self._args.n_inh))
        Wii = np.ones((self._args.n_inh, self._args.n_inh))

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
