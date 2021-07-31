import tensorflow as tf
import numpy as np
import os
import model
from tensorflow.keras.layers import Dense
from layers import Linear
from model import Model



class BaseActor:

    def __init__(self, args, rnn_params):
        self._args = args
        self._rnn_params = rnn_params

    def forward_pass(self, stimulus, h, m, gate_input=False):

        activity = []
        modulation = []
        gate = 0. if gate_input else 1.
        for stim in tf.unstack(stimulus,axis=1):
            bottom_up = gate * stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, -self._rnn_params.n_top_down:]
            if self._args.training_type=='RL':
                top_down = tf.zeros((stim.shape[0], self._rnn_params.n_top_down), dtype=tf.float32)
            output = self.model([bottom_up, top_down, h, m], training=True)
            h = output[0]
            m = output[1]
            activity.append(h)
            modulation.append(m)

        return tf.stack(activity, axis=1), tf.stack(modulation, axis=1)

    def determine_steady_state(self, stimulus):

        h, m = self.RNN.intial_activity()
        batch_size = stimulus.shape[0]
        h = tf.tile(h, (batch_size, 1))
        m = tf.tile(m, (batch_size, 1))
        activity, modulation = self.forward_pass(stimulus, h, m, gate_input=True)
        mean_activity = tf.reduce_mean(activity, axis=(0,1))
        mean_modulation = tf.reduce_mean(modulation, axis=(0,1))
        mean_activity = tf.tile(mean_activity[tf.newaxis, :], (batch_size, 1))
        modulation = tf.tile(mean_modulation[tf.newaxis, :], (batch_size, 1))

        return mean_activity, mean_modulation, activity


    def reset_optimizer(self):
        print('Resetting optimizer...')
        for var in self.opt.variables():
            print(f'Resetting {var.name}')
            var.assign(tf.zeros_like(var))



class ActorSL(BaseActor):

    def __init__(self, args, rnn_params, saved_model_path=None, learning_type='supervised'):
        self._args = args
        self._args.training_type = 'supervised'
        self._rnn_params = rnn_params
        if rnn_params is not None:
            self.RNN = Model(self._rnn_params, learning_type='supervised')
            self.model = self.RNN.model
        else:
            self.model = tf.keras.models.load_model(saved_model_path)
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=self._args.adam_epsilon)


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

        stimulus = batch[0]
        labels = batch[1]
        mask = batch[2]
        reward_matrix = batch[3]

        with tf.GradientTape(persistent=False) as tape:

            activity, policy = self.training_batch(
                                        stimulus,
                                        labels,
                                        h, m)

            policy = tf.stack(policy, axis=1)
            activity = tf.stack(activity, axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels, policy)
            loss = tf.reduce_mean(mask * loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = []
        for g, v in zip(grads, self.model.trainable_variables):
            g = tf.clip_by_norm(g, 0.5)
            #if 'top_down0' in v.name:
            #    g * 0.1


            grads_and_vars.append((g,v))

        self.opt.learning_rate.assign(learning_rate)
        self.opt.apply_gradients(grads_and_vars)

        return loss, activity, policy



class ActorRL(BaseActor):

    def __init__(self, args, rnn_params, saved_model_path=None):

        self._args = args
        self._rnn_params = rnn_params
        if rnn_params is not None:
            self.RNN = Model(self._rnn_params, learning_type='RL')
            self.model = self.RNN.model
        else:
            self.model = tf.keras.models.load_model(saved_model_path)
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-05)


    def get_actions(self, state):
        # numpy based
        h, m, policy, values = self.model.predict(state)
        policy += 1e-9
        actions = []
        for i in range(self._args.batch_size):
            try:
                action = np.random.choice(self._rnn_params.n_actions, p=policy[i,:])

            except:
                action = 0
            actions.append(action)

        log_policy = [np.log(policy[i, actions[i]]) for i in range(self._args.batch_size)]
        actions = np.stack(actions, axis = 0)
        log_policy = np.stack(log_policy, axis = 0)

        return h, m, log_policy, actions, values

    def compute_policy_loss(self, old_policy, new_policy, gaes):

        new_policy = new_policy[:, tf.newaxis]
        gaes = tf.stop_gradient(gaes)
        old_policy = tf.stop_gradient(old_policy)
        ratio = tf.math.exp(new_policy - old_policy)
        #print('ratio', ratio)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - self._args.clip_ratio, 1 + self._args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)

        return surrogate

    def train(self, states, context, h, m, actions, gaes, td_targets, old_policy, mask):

        #print('TRAIN')
        #print(actions)
        #print(actions.shape)

        actions = tf.squeeze(tf.cast(actions, tf.int32))
        actions_one_hot = tf.one_hot(actions, self._rnn_params.n_actions)


        if self._args.normalize_gae:
            gaes -= tf.reduce_mean(gaes)
            gaes /= (1e-8 + tf.math.reduce_std(gaes))
            gaes = tf.clip_by_value(gaes, -3, 3.)

        with tf.GradientTape() as tape:

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
            value_loss = self._args.critic_coeff * 0.5 * tf.reduce_mean(mask * tf.square(tf.stop_gradient(td_targets) - values))
            critic_delta = tf.reduce_mean(tf.square(values - td_targets))

            loss = policy_loss + value_loss - entropy_loss


        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, self._args.clip_grad_norm)
        grads_and_vars = []
        for g,v in zip(grads, self.model.trainable_variables):
            grads_and_vars.append((g, v))

        self.opt.apply_gradients(grads_and_vars)

        return loss, critic_delta, global_norm


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        self.set_correction_term()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def scroll_forward(self):
        for _ in range(1024):
            self.__call__()

    def set_correction_term(self):
        x = []
        for _ in range(10000):
            x.append(self.__call__())
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

    def get_actions(self, state, std):
        mu = self.model.predict(state)
        noise = self.OU()
        noise = np.clip(noise, -self._args.OU_clip_noise, self._args.OU_clip_noise)
        action = mu + std * noise / self.OU.OU_std
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, axis=-1, keepdims=True)

    def reset_param_noise(self, noise_std):

        for var in self.model.non_trainable_variables:
            print(var.name, var.shape)
            var.assign(np.random.normal(0, noise_std, size=var.shape).astype(np.float32))


    def create_model(self):
        state_input = tf.keras.Input((self.state_dim,))
        mu_output = Dense(self.action_dim, activation='linear', use_bias=True)(state_input)
        return tf.keras.models.Model(state_input, mu_output)

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-self._args.clip_ratio, 1.0+self._args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        #print('cont ratio', ratio)
        return surrogate



    def train(self, states, actions, gaes, log_old_policy, mask, std):

        std = tf.cast(std, tf.float32)
        if self._args.normalize_gae_cont:
            gaes -= tf.reduce_mean(gaes)
            gaes /= (1e-8 + tf.math.reduce_std(gaes))
            gaes = tf.clip_by_value(gaes, -3, 3.)

        with tf.GradientTape() as tape:
            mu = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = tf.reduce_mean(mask * self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes))
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, self._args.clip_grad_norm)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, global_norm
