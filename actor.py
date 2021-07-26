import tensorflow as tf
import numpy as np
import os
import model
from tensorflow.keras.layers import Dense
from layers import Linear
from model import Model
#import losses



class BaseActor:

    def __init__(self, args, rnn_params):
        self._args = args
        self._rnn_params = rnn_params

    def forward_pass(self, stimulus, h, m, gate_input=False):

        gate = 0. if gate_input else 1.
        activity = []
        modulation = []
        for stim in tf.unstack(stimulus,axis=1):
            bottom_up = gate * stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, self._rnn_params.n_bottom_up:]
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
        mean_activity = tf.reduce_mean(activity[:, -10:, :], axis=1)
        mean_modulation = tf.reduce_mean(modulation[:, -10:, :], axis=1)

        return mean_activity, mean_modulation, activity


    def reset_optimizer(self):
        print('Resetting optimizer...')
        for var in self.opt.variables():
            print(f'Resetting {var.name}')
            var.assign(tf.zeros_like(var))



class ActorSL(BaseActor):

    def __init__(self, args, rnn_params, saved_model_path=None, learning_type='supervised'):
        self._args = args
        self._rnn_params = rnn_params
        if rnn_params is not None:
            self.RNN = Model(self._rnn_params, learning_type='supervised')
            self.model = self.RNN.model
        else:
            self.model = tf.keras.models.load_model(saved_model_path)
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-07)


    def training_batch(self, stimulus, labels, h, m):

        policy = []
        activity = []

        for stim in tf.unstack(stimulus,axis=1):
            bottom_up = stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, self._rnn_params.n_bottom_up:]
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
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-07)


    def get_actions(self, state):
        # numpy based
        x = state[0][:,:self._rnn_params.n_bottom_up]
        y = state[0][:,self._rnn_params.n_bottom_up:]
        h = state[1]
        m = state[2]
        h, m, policy, values = self.model.predict([x, y, h, m])
        actions = []
        for i in range(self._args.batch_size):
            try:
                action = np.random.choice(self.action_dim, p=policy[i,:])
            except:
                action = 0
            actions.append(action)

        log_policy = [np.log(1e-8 + policy[i, actions[i]]) for i in range(self._args.batch_size)]
        actions = np.stack(actions, axis = 0)
        log_policy = np.stack(log_policy, axis = 0)

        return h, m, log_policy, actions, values


    def training_batch(self, stimulus, labels, mask, reward_matrix, h, m, RL=False):

        policy = []
        actions = []
        activity = []
        pred_vals = []

        for stim, reward_mat in zip(tf.unstack(stimulus,axis=1), tf.unstack(reward_matrix,axis=1)):
            bottom_up = stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, self._rnn_params.n_bottom_up:]
            p, h, m = self.model([bottom_up, top_down, h, m], training=True)
            policy.append(p)
            activity.append(h)
            #pred_vals.append(c)
            if RL:
                #print('CT', continue_trial.shape)
                action = tf.random.categorical(p, 1)
                action_one_hot = tf.squeeze(tf.one_hot(action, self._rnn_params.n_actions))
                actions.append(action_one_hot)


        return policy, activity, pred_vals, actions



    def train(self, batch, h, m, learning_rate):

        if self._args.training_type == 'supervised':
            return self.train_SL(batch, h, m, learning_rate)
        elif self._args.training_type == 'actor_critic':
            return self.train_RL(batch, h, m, learning_rate)



    def train_RL(self, batch, h, m, learning_rate):

        stimulus = batch[0]
        mask = batch[2]
        reward_matrix = batch[3]

        with tf.GradientTape(persistent=False) as tape: #not sure persistent matters
            policy, activity, pred_vals, actions = self.training_batch(
                                                            stimulus,
                                                            None,
                                                            mask,
                                                            reward_matrix,
                                                            h, m,
                                                            RL=True)
            policy = tf.stack(policy, axis=1)
            pred_vals = tf.stack(pred_vals, axis=1)
            activity = tf.stack(activity, axis=1)
            actions = tf.stop_gradient(tf.stack(actions, axis=1))

            rewards, mask = losses.calculate_reward(actions, reward_matrix, mask)

            loss, metrics = losses.actor_critic(
                               policy,
                               actions,
                               pred_vals,
                               rewards,
                               mask,
                               value_loss_factor=0.01,
                               entropy_loss_factor=0.001,
                               discount_factor=0.9)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)

        self.opt.learning_rate.assign(learning_rate)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, activity, policy, rewards, mask

    def train_SL(self, batch, h, m, learning_rate):

        stimulus = batch[0]
        labels = batch[1]
        mask = batch[2]
        reward_matrix = batch[3]

        with tf.GradientTape(persistent=False) as tape:

            policy, activity, _, _ = self.training_batch(
                                            stimulus,
                                            labels,
                                            mask,
                                            reward_matrix,
                                            h, m,
                                            RL=False)

            policy = tf.stack(policy, axis=1)
            activity = tf.stack(activity, axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels, policy)
            loss = tf.reduce_mean(mask * loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        grads_and_vars = []
        for g, v in zip(grads, self.model.trainable_variables):
            g = tf.clip_by_norm(g, 0.5)
            grads_and_vars.append((g,v))

        self.opt.learning_rate.assign(learning_rate)
        self.opt.apply_gradients(grads_and_vars)

        return loss, activity, policy, -1, -1

    def gae_target(self, rewards, values, last_value, done):

        gae = np.zeros(rewards.shape, dtype=np.float32)
        gae_cumulative = 0.
        nsteps = rewards.shape[1]
        for k in reversed(range(nsteps)):
            if k == nsteps - 1:
                nextnonterminal = 1.0 - done[:,-1]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - done[:, k+1]
                nextvalues = values[:, k+1]
            delta = rewards[:, k] + self._args.gamma * nextvalues * nextnonterminal - values[:, k]
            gae_cumulative = self._args.gamma * self._args.lmbda * gae_cumulative * nextnonterminal + delta
            gae[:, k] = gae_cumulative
        target_value = gae + values

        #gae -= tf.reduce_mean(gae)
        #gae /= (1e-4 + tf.math.reduce_std(gae))

        return gae, target_value



class ActorContinuousRL:
    def __init__(self, args, state_dim, action_dim, action_bound, std_bound):
        self._args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = 2.
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.cont_learning_rate, epsilon=1e-05)

    def get_action(self, state, std):
        mu, _ = self.model.predict(state)
        action = np.random.normal(mu, std)
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def create_model(self):
        state_input = tf.keras.Input((self.state_dim,))
        mu_output = Dense(self.action_dim, activation='linear', use_bias=False)(state_input)
        std_output = Dense(self.action_dim, activation='softplus', use_bias=False, trainable=False)(state_input) # DON'T USE THIS FOR NOW
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        gaes = gaes[:, tf.newaxis]
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-args.cont_clip_ratio, 1.0+args.cont_clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)

        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes, std):

        if self._args.normalize_gae:
            gaes -= tf.reduce_mean(gaes)
            gaes /= (1e-8 + tf.math.reduce_std(gaes))
            gaes = tf.clip_by_value(gaes, -3, 3.)

        with tf.GradientTape() as tape:
            mu, _ = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
