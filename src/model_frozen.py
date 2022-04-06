import numpy as np
import copy
from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt
from .layers_frozen import Linear, Recurrent, Evolve
from . import util

class Model():

    def __init__(self, args, n_RFs=2, **kwargs):
        self._args = args
        self.n_RFs = n_RFs
        self.max_weight_value = 2.
        self._set_pref_dirs()
        self.model = self.create_network()


    def initialize_weights(self):

        ######################################################################## 
        # Set up for generating weight matrices (establish E/I identity, etc.)
        self.EI_list = np.array([1. for i in range(self._args.n_exc)] + \
            [-1. for i in range(self._args.n_inh)]).astype(np.float32)

        self.EI_matrix = tf.linalg.diag(tf.constant(self.EI_list))
        self.mask = tf.ones_like(self.EI_matrix) - \
            tf.linalg.diag(tf.ones(self._args.n_hidden))
        self.EI = tf.cast(self.EI_matrix, tf.float32)
        self.mask = tf.cast(self.mask, tf.float32)

        ws = {}
        ws['w_bottom_up'] = self.initialize_bottom_up_weights()
        ws['w_policy'], ws['w_critic'] = self.initialize_output_weights()
        ws['w_top_down1'] = self.initialize_top_down_weights()
        ws['w_rnn'], ws['b_rnn'] = self.initialize_recurrent_weights()
        ws['w_mod'] = self.initialize_modulation_weights()
        ws['alpha_soma'], ws['alpha_modulator'] = \
            self.initialize_decay_time_constants()

        return ws

    def define_model_logic(self, ws):

        ########################################################################
        # 1. Set up inputs to the network
        x_input = tf.keras.Input((self._args.n_bottom_up,)) # Bottom-up input
        h_input = tf.keras.Input((self._args.n_hidden,)) # Prev. activity
        m_input = tf.keras.Input((self._args.n_hidden,)) # Prev. activity mod.
        y_input = tf.keras.Input((self._args.n_hidden,)) # Top-down input
            
        ########################################################################    
        # 2. Define currents that neurons receive (bottom-up, recurrent, 
        # modulatory)
        bottom_up_current = Linear(
                            ws['w_bottom_up'],
                            name='bottom_up')(x_input)

        rec_current = Recurrent(
                        ws['w_rnn'],
                        ws['b_rnn'],
                        self.mask,
                        self.EI_matrix,
                        name='rnn')(h_input)

        modulation = Linear(
                        ws['w_mod'],
                        name='modulation')(h_input)

        ######################################################################## 
        # 4. Update modulation for current timestep
        modulator = Evolve(ws['alpha_modulator'], trainable=False, 
            name='alpha_modulator')((m_input, modulation))
        effective_mod = 1 / (1 + tf.nn.relu(modulator))

        ######################################################################## 
        # 5. Generate recurrent network noise
        noise_rnn_sd = np.sqrt(2/ws['alpha_soma'])*self._args.noise_rnn_sd
        noise = tf.random.normal(tf.shape(h_input), 
            mean=0., stddev=noise_rnn_sd)

        ######################################################################## 
        # 6. Integrate all of these current sources (top-down, bottom-up, 
        # recurrent, modulatory, noise) at soma, and apply nonlinearity
        soma_input = bottom_up_current + rec_current * effective_mod + noise \
            + y_input
        h = Evolve(ws['alpha_soma'], trainable=False,
            name='alpha_soma')((h_input, soma_input))
        h = tf.nn.relu(h)
        h_out = h[..., :self._args.n_exc] if self._args.restrict_output_to_exc \
            else h

        ######################################################################## 
        # 7. Generate network output (linear fxn of hidden activity vector)
        policy = Linear(
                    ws['w_policy'],
                    bias=True,
                    name='policy')(h_out)


        ######################################################################## 
        # 8. Return different model types for RL vs. SL
        return tf.keras.models.Model(
            inputs=[x_input, y_input, h_input, m_input],
            outputs=[h, modulator, policy])

        return None

    def create_network(self):
        ws    = self.initialize_weights()
        model = self.define_model_logic(ws)
        return model

    def generate_new_weights(self, new_args, random_seed=1):
        # Draw new weights given the supplied hyperparams

        # Set random seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        self._args = new_args
        w_bottom_up = self.initialize_bottom_up_weights()

        w_policy, w_critic = self.initialize_output_weights()
        w_top_down1 = self.initialize_top_down_weights()
        
        w_rnn, b_rnn = self.initialize_recurrent_weights()
        w_mod = self.initialize_modulation_weights()

        alpha_soma, alpha_modulator = self.initialize_decay_time_constants()
        b_pol = np.float32(np.zeros((1,self._args.n_actions)))
        b_pol[:,0] = 1.

        var_reset_dict = {'modulation_w:0': w_mod, 
                          'bottom_up_w:0' : w_bottom_up,
                          'rnn_w:0'       : w_rnn, 
                          'rnn_b:0'       : b_rnn, 
                          'w_top_down1:0' : w_top_down1,  
                          'policy_w:0'    : w_policy, 
                          'policy_b:0'    : b_pol,
                          'critic_w:0'    : w_critic, 
                          'critic_b:0'    : None,
                          'alpha_modulator:0': alpha_modulator, 
                          'alpha_soma:0'     : alpha_soma}

        allvar = self.model.non_trainable_variables + \
            self.model.trainable_variables

        for var in allvar:
            for key, reset_val in var_reset_dict.items():
                if var.name == key:
                    print(f'Resetting {var.name}', reset_val.shape)
                    if reset_val is not None:
                        var.assign(reset_val)
                    else:
                        var.assign(tf.zeros_like(var))


    def initial_activity(self):

        soma_exc = tf.random.uniform((1, self._args.n_exc), 
            0., self._args.init_exc)
        soma_inh = tf.random.uniform((1, self._args.n_inh), 
            0., self._args.init_inh)
        soma_init = tf.concat((soma_exc, soma_inh), axis = -1)
        mod_init = tf.random.uniform((1, self._args.n_hidden), 
            0., self._args.init_mod)

        return soma_init, mod_init


    def _set_pref_dirs(self):

        '''Motion direcion input neurons will have prefered direction, but
        fixation neurons will not. Fixation neurons will be at the end of
        the array'''
        N = self._args.n_bottom_up - self._args.n_motion_tuned

        # 1. Establish spatial topology in RNN hidden connectivity
        rnn_phase = np.linspace(0, 2*np.pi, self._args.n_hidden)
        rnn_phase0 = rnn_phase[np.random.permutation(self._args.n_hidden)]
        rnn_phase1 = rnn_phase[np.random.permutation(self._args.n_hidden)]

        # 2. Add spatial topology w.r.t. bottom-up inputs into RNN
        if self.n_RFs == 2 and self._args.bottom_up_topology:
            # Two RFs
            motion_phase = np.linspace(0, 2*np.pi, self._args.n_motion_tuned//2)
            motion_phase = np.concatenate((motion_phase, motion_phase), axis=-1)
            RF_phase = self._args.n_motion_tuned//2 * [0] + \
                self._args.n_motion_tuned//2 * [np.pi]
            RF_phase = np.array(RF_phase)
            
            RF_rnn = np.cos(RF_phase[:, np.newaxis] - rnn_phase1[np.newaxis,:])
            RF_rnn = np.vstack((RF_rnn, np.zeros((N, self._args.n_hidden))))

        elif self._args.n_RFs <= 1 or not self._args.bottom_up_topology:
            # One RF
            motion_phase = np.linspace(0, 2*np.pi, self._args.n_motion_tuned)
            RF_rnn =  np.zeros((self._args.n_bottom_up, self._args.n_hidden))
        elif self._args.n_cue_tuned > 2:
            assert False, "Not sure how to handle 3 or more cue tuned neurons"

        # 3. Establish topology w.r.t. rule projection into RNN
        tdh = self._args.n_top_down_hidden
        td_phase = np.linspace(0, 2*np.pi, tdh)
        td_phase0 = td_phase[np.random.permutation(tdh)]
        td_phase1 = td_phase[np.random.permutation(tdh)]

        motion_rnn = np.cos(
            motion_phase[:, np.newaxis] - rnn_phase0[np.newaxis, :])
        motion_rnn = np.vstack((motion_rnn, np.zeros((N, self._args.n_hidden))))

        # 4. Use these phases w.r.t. RNN connectivity, bottom-up input,
        # and top-down input to generate overall phase values
        self._inp_rnn_phase = 0.5 * motion_rnn + 0.5 * RF_rnn

        self._rnn_rnn_phase = \
            0.5*np.cos(rnn_phase0[:,np.newaxis]-rnn_phase0[np.newaxis,:]) + \
            0.5*np.cos(rnn_phase1[:,np.newaxis]-rnn_phase1[np.newaxis,:])
        self._td_rnn_phase = \
            0.5*np.cos(td_phase0[:,np.newaxis]-rnn_phase0[np.newaxis,:]) + \
            0.5*np.cos(td_phase1[:,np.newaxis]-rnn_phase1[np.newaxis,:])


    def initialize_decay_time_constants(self):
        # Initialize time constants of decay of network activity + modulation.

        alpha_soma = np.clip(self._args.dt/self._args.tc_soma, 0., 1.)
        alpha_modulator = np.clip(self._args.dt/self._args.tc_modulator, 0, 1.)
        return np.float32(alpha_soma), np.float32(alpha_modulator)

    def initialize_output_weights(self):
        # Initialize weights to outputs.

        initializer = tf.keras.initializers.GlorotNormal()
        if self._args.restrict_output_to_exc:
            w_policy = initializer(shape=(self._args.n_exc,
                self._args.n_actions))
            w_critic = initializer(shape=(self._args.n_exc, 2))
        else:
            w_policy = initializer(shape=(self._args.n_hidden, 
                self._args.n_actions))
            w_critic = initializer(shape=(self._args.n_hidden, 2))

        return tf.cast(w_policy, tf.float32), tf.cast(w_critic, tf.float32)

    def initialize_top_down_weights(self):
        # Initialize top-down weights.
        
        # Weights for fixed top-down projection
        We = np.random.gamma(
                        self._args.td_E_kappa,
                        1.,
                        size = (self._args.n_top_down_hidden, self._args.n_exc))
        Wi = np.random.gamma(
                        self._args.td_I_kappa,
                        1.,
                        size = (self._args.n_top_down_hidden, self._args.n_inh))

        We *= util.von_mises(
                self._td_rnn_phase[:, :self._args.n_exc],
                self._args.td_E_topo)
        Wi *= util.von_mises(
                self._td_rnn_phase[:, self._args.n_exc:],
                self._args.td_I_topo)

        # 3. Scale by top-down weight value + allow projection only to even-
        # numbered neurons.
        w1 = self._args.td_weight * np.concatenate((We, Wi), axis=1)
        w1 = np.clip(w1, 0., self.max_weight_value)
        w1[:, 1::2] = 0.

        return tf.cast(w1, tf.float32)

    def initialize_bottom_up_weights(self):

        '''Input neurons will project unfiformly to the half the EXC neurons;
        input strength determined by Von Mises distribution.
        Inputs only project to every second neuron'''

        # 1. Draw weights onto E vs. I units in RNN separately,
        # from gamma distribution, w/ shape K determined sep. for each
        We = np.random.gamma(
                        self._args.inp_E_kappa,
                        1.,
                        size = (self._args.n_bottom_up, self._args.n_exc))
        Wi = np.random.gamma(
                        self._args.inp_I_kappa,
                        1.,
                        size = (self._args.n_bottom_up, self._args.n_inh))

        # 2. Build motion tuning also into projection into hidden layer: 
        # e.g. the similarity of projection of any pair of motion-tuned
        # neurons onto any single RNN neuron is related to the similarity
        # of those motion neurons' tuning
        We[:, :] *= util.von_mises(
                    self._inp_rnn_phase[:, :self._args.n_exc],
                    self._args.inp_E_topo)
        Wi[:, :] *= util.von_mises(
                    self._inp_rnn_phase[:, self._args.n_exc:],
                    self._args.inp_I_topo)

        # 3. Modulate overall strength of input projection by hyperparameter 
        # controlling input strength, and allow projection only to odd-numbered
        # units.
        W = np.concatenate((We, Wi), axis=1)
        W *= self._args.input_weight
        W = np.clip(W, 0., self.max_weight_value)
        W[:, ::2] = 0.


        return np.float32(W)

    def initialize_modulation_weights(self):
        # Initialize modulation weights -- e.g. the weights that tamp
        # down activity when it grows too high.

        # 1. Determine strength of modulation for all E/E, E/I etc. pairings
        beta_EE = self._args.mod_EE_weight / self._args.n_hidden
        beta_EI = self._args.mod_EI_weight / self._args.n_hidden
        beta_IE = self._args.mod_IE_weight / self._args.n_hidden
        beta_II = self._args.mod_II_weight / self._args.n_hidden

        # 2. Apply these weights uniformly to all of the relevant synapse groups
        Wee = np.ones((self._args.n_exc, self._args.n_exc))
        Wei = np.ones((self._args.n_inh, self._args.n_exc))
        Wie = np.ones((self._args.n_exc, self._args.n_inh))
        Wii = np.ones((self._args.n_inh, self._args.n_inh))

        We = np.hstack((beta_EE * Wee, beta_IE * Wie))
        Wi = np.hstack((beta_EI * Wei, beta_II * Wii))
        W = np.vstack((We, Wi))

        return np.float32(W)


    def initialize_recurrent_weights(self):
        # Initialize recurrent weights.

        # 1. Draw weights from gamma distrbution,
        # w/ shape specific to each synapse group
        Wee = np.random.gamma(self._args.EE_kappa, 1., 
            size = (self._args.n_exc, self._args.n_exc))
        Wie = np.random.gamma(self._args.IE_kappa, 1., 
            size = (self._args.n_exc, self._args.n_inh))
        Wei = np.random.gamma(self._args.EI_kappa, 1., 
            size = (self._args.n_inh, self._args.n_exc))
        Wii = np.random.gamma(self._args.II_kappa, 1., 
            size = (self._args.n_inh, self._args.n_inh))

        # 2. Control the strength of the reciprocal connection
        # by adding in scaled copy of the transpose (idea: 
        # if reciprocal connectiviity is high, then adding 
        # transpose multiplied by high constant specifically 
        # enriches mutual connectivity between pairs). Scaled down
        # by factor of 1 / (1 + alpha)
        Wie_temp = copy.copy(Wie)
        Wie += (self._args.alpha_EI * np.transpose(Wei))
        Wei += (self._args.alpha_EI * np.transpose(Wie_temp))
        Wee += (self._args.alpha_EE * np.transpose(Wee))
        Wii += (self._args.alpha_II * np.transpose(Wii))
        Wee /= (1 + self._args.alpha_EE)
        Wie /= (1 + self._args.alpha_EI)
        Wei /= (1 + self._args.alpha_EI)
        Wii /= (1 + self._args.alpha_II)

        # 3. Applying the topological modifier (idea is like geometric connect.
        # law: stronger connections for units that are closer by.)
        Wee *= util.von_mises(
                    self._rnn_rnn_phase[:self._args.n_exc, :self._args.n_exc],
                    self._args.EE_topo)
        Wei *= util.von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, :self._args.n_exc],
                    self._args.EI_topo)
        Wie *= util.von_mises(
                    self._rnn_rnn_phase[:self._args.n_exc, self._args.n_exc:],
                    self._args.IE_topo)
        Wii *= util.von_mises(
                    self._rnn_rnn_phase[self._args.n_exc:, self._args.n_exc:],
                    self._args.II_topo)

        # 4. Pull all matrices together, scale by rnn_weight hyperparameter,
        # and zero out all self-connections
        We = np.hstack((Wee, Wie))
        Wi = np.hstack((Wei, Wii))
        w_rnn = np.vstack((We, Wi))
        w_rnn *= self._args.rnn_weight

        b_rnn = np.zeros((self._args.n_hidden), dtype = np.float32)

        for i in range(self._args.n_hidden):
            w_rnn[i, i] = 0.

        w_rnn = np.clip(w_rnn, 0., self.max_weight_value)

        return np.float32(w_rnn), np.float32(b_rnn)
