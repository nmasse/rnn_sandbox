import tensorflow as tf
import numpy as np
import os, copy
from tensorflow.keras.layers import Dense
from . import model, model_experimental, model_LSTM
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from .DNI import DNI

class BaseActor:

    def __init__(self, args, rnn_params):
        self._args = args
        self._rnn_params = rnn_params

    def forward_pass(self, stimulus, h, m):

        activity = []
        modulation = []
        for stim in tf.unstack(stimulus,axis=1):
            bottom_up = stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, -self._rnn_params.n_top_down:]
            if self._args.training_type=='RL':
                top_down = tf.zeros((stim.shape[0], self._rnn_params.n_top_down), dtype=tf.float32)
            output = self.model([bottom_up, top_down, h, m], training=False)
            if self._args.training_alg == 'DNI':
                h = output[0]
                m = output[2]
            else:
                h = output[0]
                m = output[1]
            activity.append(h)
            modulation.append(m)

        return tf.stack(activity, axis=1), tf.stack(modulation, axis=1)

    def determine_steady_state(self, stimulus):

        t0 = self._args.steady_state_start // self._rnn_params.dt
        t1 = self._args.steady_state_end // self._rnn_params.dt

        h, m = self.RNN.initial_activity()
        batch_size = stimulus.shape[0]
        h = tf.tile(h, (batch_size, 1))
        m = tf.tile(m, (batch_size, 1))
        activity, modulation = self.forward_pass(stimulus, h, m)
        mean_activity = tf.reduce_mean(activity[:,t0:t1,:], axis=(0,1))
        mean_modulation = tf.reduce_mean(modulation[:,t0:t1,:], axis=(0,1))
        mean_activity = tf.tile(mean_activity[tf.newaxis, :], (batch_size, 1))
        modulation = tf.tile(mean_modulation[tf.newaxis, :], (batch_size, 1))

        return mean_activity, mean_modulation, activity

    def reset_optimizer(self):
        print('Resetting optimizer...')
        for var in self.opt.variables():
            print(f'Resetting {var.name}')
            var.assign(tf.zeros_like(var))

class ActorSL(BaseActor):

    def __init__(self, args, rnn_params, saved_model_path=None, 
        learning_type='supervised', n_RFs=2):
        self._args = args
        self._args.training_type = 'supervised'
        self._rnn_params = rnn_params
        if rnn_params is not None:
            m = eval(f"{args.model_type}.Model")
            self.RNN = m(self._rnn_params, learning_type='supervised', n_RFs=n_RFs,
                training_alg=self._rnn_params.training_alg)
            self.model = self.RNN.model
        else:
            self.model = tf.keras.models.load_model(saved_model_path)
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, 
            epsilon=self._args.adam_epsilon)

        # Create DNI object
        weights = self.model.get_weights()
        names = [weight.name for layer in self.model.layers for weight in layer.weights]
        weights = dict(zip(names, weights))

        if self._args.training_alg == 'DNI':
            self.dni_opt = tf.keras.optimizers.Adam(1e-3)

            self.DNI = DNI(self._rnn_params, 
                weights['top_down1_w:0'], 
                self.dni_opt, 
                weights['rnn_w:0'] * self.RNN.EI_matrix, 
                weights['alpha_soma:0'],
                batch_size=self._args.batch_size)


    def training_batch(self, stimulus, labels, h, m):

        policy = []
        activity = []

        for stim in tf.unstack(stimulus,axis=1):
            bottom_up = stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, -self._rnn_params.n_top_down:]
            h, m, p = self.model([bottom_up, top_down, h, m], training=True)
            policy.append(p)
            activity.append(h)

        return activity, policy

    def train(self, batch, h, m, learning_rate):

        if self._args.training_alg == 'DNI':
            return self.trainDNI(batch, h, m, learning_rate)
        else:
            return self.trainBPTT(batch, h, m, learning_rate)

    def trainDNI(self, batch, h, m, learning_rate):

        # Unpack necessary information
        stimulus      = batch[0]
        labels        = batch[1]
        mask          = batch[2]
        reward_matrix = batch[3]
        h_prev        = copy.copy(h)


        policy   = np.zeros((stimulus.shape[0], stimulus.shape[1], 
            reward_matrix.shape[-1]))
        activity = np.zeros((stimulus.shape[0], stimulus.shape[1], 
            h.shape[-1]))

        # Reset information from the last batch of trials
        self.DNI.reset_learning()

        m_next = m.numpy().copy()

        # Loop through batch of trials, one timestep at a time, and perform DNI
        for j, t in enumerate(tf.unstack(stimulus, axis=1)): 

            # Separate bottom-up and top-down inputs
            bu = t[:, :self._rnn_params.n_bottom_up]
            td = t[:, -self._rnn_params.n_top_down:]
            lab = labels[:,j]
            if j > 0:
                lab_prev = labels[:,j-1]
            else:
                lab_prev = lab
            
            # Run the model on the inputs for this timestep
            h_next, u, m_next, pol_next, td_h, td_c = \
                self.model([bu, td, h, m], training=False)

            # Store activity/policy
            activity[:,j,:] = h_next.numpy()
            policy[:,j,:]   = pol_next.numpy()

            # If first iter: don't make update
            if j == 0:
                pol = copy.copy(pol_next)

            # Pass to DNI, retrieve and apply gradients
            else:
                grads = self.DNI(tf.squeeze(h_next), 
                    tf.squeeze(h),
                    tf.squeeze(pol_next),
                    tf.squeeze(pol),
                    tf.squeeze(lab), 
                    tf.squeeze(lab_prev),
                    tf.squeeze(u), 
                    tf.squeeze(td),
                    tf.squeeze(mask[:,j]))

                grads_and_vars = []
                g_and_tv = zip(grads, self.model.trainable_variables)
                for i, (g, v) in enumerate(g_and_tv):
                    g *= mask[:,j].mean()
                    g /= self._args.batch_size # because sum taken in DNI over axis 0
                    g = tf.cast(tf.clip_by_norm(g, self._args.clip_grad_norm), 
                        tf.float32)

                    grads_and_vars.append((g,v))

                self.opt.learning_rate.assign(learning_rate)
                if learning_rate > 0:
                    self.opt.apply_gradients(grads_and_vars)

            # Update h_i and m_i w/ new values
            h = h_next
            m = m_next
            pol = pol_next

        # Compute the loss on the whole batch (after the fact)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, policy)
        loss = tf.reduce_mean(mask * loss)

        return loss, activity, policy


    def trainBPTT(self, batch, h, m, learning_rate):

        # Unpack relevant information
        stimulus      = batch[0]
        labels        = batch[1]
        mask          = batch[2]
        reward_matrix = batch[3]

        # Train via BPTT, using TF autograd
        with tf.GradientTape(persistent=False) as tape:

            activity, policy = self.training_batch(
                                        stimulus,
                                        labels,
                                        h, m)

            policy = tf.stack(policy, axis=1)
            activity = tf.stack(activity, axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels, 
                policy)
            loss = tf.reduce_mean(mask * loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = []
        for g, v in zip(grads, self.model.trainable_variables):
            g = tf.clip_by_norm(g, self._args.clip_grad_norm)
            if "top_down" in v.name:
                g *= self._args.top_down_grad_multiplier
            grads_and_vars.append((g,v))

        self.opt.learning_rate.assign(learning_rate)
        if learning_rate > 0:
            self.opt.apply_gradients(grads_and_vars)

        return loss, activity, policy


class ActorRL(BaseActor):

    def __init__(self, args, rnn_params, saved_model_path=None, n_RFs=2):

        self._args = args
        self._rnn_params = rnn_params
        if rnn_params is not None:
            m = eval(f"{args.model_type}.Model")
            self.RNN = m(self._rnn_params, learning_type='RL', n_RFs=n_RFs)
            self.model = self.RNN.model
        else:
            self.model = tf.keras.models.load_model(saved_model_path)

        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-05)


    def get_actions(self, state, do_return_log_policy=True):
        # numpy based
        if self._args.model_type == 'model_LSTM':
            state[2] = np.expand_dims(state[2], 1)

        h, m, policy, values = self.model.predict(state)
        policy += 1e-9
        actions = []
        for i in range(self._args.batch_size):
            try:
                action = np.random.choice(self._rnn_params.n_actions, p=policy[i,:])

            except:
                action = 0
            actions.append(action)

        if do_return_log_policy:
            log_policy = [np.log(policy[i, actions[i]]) for i in range(self._args.batch_size)]
            log_policy = np.stack(log_policy, axis = 0)
        else:
            log_policy = None
        actions = np.stack(actions, axis = 0)
        

        return h, m, log_policy, actions, values

    def compute_policy_loss(self, old_policy, new_policy, gaes):

        new_policy = new_policy[:, tf.newaxis]
        gaes = tf.stop_gradient(gaes)
        old_policy = tf.stop_gradient(old_policy)
        ratio = tf.math.exp(new_policy - old_policy)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - self._args.clip_ratio, 1 + self._args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)

        return surrogate

    def train(self, states, context, h, m, actions, gaes, td_targets, 
        old_policy, mask, learning_rate):

        actions = tf.squeeze(tf.cast(actions, tf.int32))
        actions_one_hot = tf.one_hot(actions, self._rnn_params.n_actions)

        with tf.GradientTape() as tape:

            if self._args.model_type == 'model_LSTM':
                h = np.expand_dims(h, 1)

            _, _, policy, values = self.model([
                                        states,
                                        context,
                                        tf.stop_gradient(h),
                                        tf.stop_gradient(m)])

            policy += 1e-9
            log_policy = tf.reduce_sum(actions_one_hot * tf.math.log(policy),axis=-1)
            entropy = - tf.reduce_sum(policy * tf.math.log(policy), axis=-1)
            surrogate = self.compute_policy_loss(old_policy, log_policy, gaes)
            policy_loss = tf.reduce_mean(mask * surrogate)

            entropy_loss = self._args.entropy_coeff * tf.reduce_mean(mask * entropy)
            value_loss = self._args.critic_coeff * 0.5 * tf.reduce_mean(mask * 
                tf.square(tf.stop_gradient(td_targets) - values))
            loss = policy_loss + value_loss - entropy_loss


        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, self._args.clip_grad_norm)
        self.opt.learning_rate.assign(learning_rate)
        if learning_rate > 0:
            if not tf.math.is_nan(global_norm):
                self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, value_loss, global_norm


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, noise_bound=3., x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.noise_bound = noise_bound
        self.reset()
        self.set_correction_term()

    def step(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def get_noise(self):
        x = self.step()
        x = x / self.OU_std
        return np.clip(x, -self.noise_bound, self.noise_bound)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def scroll_forward(self):
        for _ in range(1024):
            self.step()

    def set_correction_term(self):
        x = []
        for _ in range(10000):
            x.append(self.step())
        self.OU_std = np.std(x)


    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)



class ActorContinuousRL:
    def __init__(self, args, state_dim, action_dim, action_bound):
        self._args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.cont_learning_rate, epsilon=1e-05)

        self.OU = OrnsteinUhlenbeckActionNoise(
            np.zeros((action_dim),dtype=np.float32),
            np.ones((action_dim),dtype=np.float32),
            theta=self._args.OU_theta)

    def get_actions(self, state, std, noise_clip=3.):
        mu = self.model.predict(state)
        if self._args.OU_noise:
            noise = self.OU.get_noise()
        else:
            noise = np.random.normal(0., 1., mu.shape)
            noise = np.clip(noise, -noise_clip, noise_clip) # Noise clip default: 3

        action = mu + std * noise
        action = np.clip(action, -self.action_bound, self.action_bound)

        # Obtain log pdf
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    
    def log_pdf(self, mu, std, action):
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, axis=-1, keepdims=True)

    def create_model(self):
        state_input = tf.keras.Input((self.state_dim,))
        mu_output = Dense(self.action_dim, use_bias=True)(state_input)
        return tf.keras.models.Model(state_input, [mu_output])

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-self._args.clip_ratio, 1.0+self._args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return surrogate

    def train(self, states, actions, gaes, log_old_policy, mask, std, learning_rate):

        std = tf.cast(std, tf.float32)
        gaes = tf.cast(gaes, tf.float32)

        with tf.GradientTape() as tape:
            mu = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            if mask is None:
                mask = 1.
            loss = tf.reduce_mean(mask * self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes))
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, self._args.clip_grad_norm_cont)
        self.opt.learning_rate.assign(learning_rate)
        if learning_rate > 0:
            if not tf.math.is_nan(global_norm):
                self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, global_norm
