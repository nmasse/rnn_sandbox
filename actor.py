import tensorflow as tf
import numpy as np
import os
import layers
from tensorflow.keras.layers import Dense


class Actor:
    def __init__(self, args, rnn_params, saved_model_path=None, learning_type='supervised'):
        self._args = args
        self._rnn_params = rnn_params
        self.learning_type = learning_type
        if rnn_params is not None:
            self.create_model()
        else:
            self.model = tf.keras.models.load_model(saved_model_path)
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-07)


    def create_model(self):

        x_input = tf.keras.Input((self._rnn_params.n_bottom_up,)) # Bottom-up input
        y_input = tf.keras.Input((self._rnn_params.n_top_down,)) # Top-down input
        h_input = tf.keras.Input((self._rnn_params.n_hidden,)) # Previous activity
        m_input = tf.keras.Input((self._rnn_params.n_hidden,)) # Previous modulation

        self.RNN = layers.Model(self._rnn_params)
        h, m = self.RNN.model((x_input, y_input, h_input, m_input))
        policy = Dense(self._rnn_params.n_actions, activation='linear', name='policy')(h)
        self.model = tf.keras.models.Model([x_input, y_input, h_input, m_input], [policy, h, m])


    def get_action(self, state):
        # numpy based
        policy, soma, modulator = self.rnn.model.predict(state)
        actions = []
        for i in range(self._args.batch_size):
            try:
                action = np.random.choice(self.action_dim, p=policy[i,:])
            except:
                action = 0
            actions.append(action)
        actions = np.stack(actions, axis = 0)

        return actions, policy, critic, soma, modulator

    def determine_steady_state(self, batch):

        h, m = self.RNN.intial_activity(self._args.batch_size)
        stimulus = batch[0]

        activity = []
        mod = []

        for stim in tf.unstack(stimulus,axis=1):
            bottom_up = 0 * stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, self._rnn_params.n_bottom_up:]
            _, h, m = self.model([bottom_up, top_down, h, m], training=True)
            activity.append(h)
            mod.append(m)

        activity = tf.stack(activity, axis=1)
        mod = tf.stack(mod, axis=1)

        mean_activity = tf.reduce_mean(activity[:, -5:, :], axis=1)
        mean_mod = tf.reduce_mean(mod[:, -5:, :], axis=1)

        return mean_activity, mean_mod, activity

    def run_batch(self, batch, h, m):

        stimulus = batch[0]
        activity = []

        for stim in tf.unstack(stimulus,axis=1):
            bottom_up = stim[:, :self._rnn_params.n_bottom_up]
            top_down = stim[:, self._rnn_params.n_bottom_up:]
            _, h, m = self.model([bottom_up, top_down, h, m], training=True)
            activity.append(h)

        return tf.stack(activity, axis=1)

    def reset_optimizer(self):

        for var in self.opt.variables():
            var.assign(tf.zeros_like(var))

    def train(self, batch, h, m, learning_rate):

        stimulus = batch[0]
        labels = batch[1]
        mask = batch[2]

        with tf.GradientTape(persistent=False) as tape: #not sure persistent matters

            policy = []
            activity = []
            for stim in tf.unstack(stimulus,axis=1):
                bottom_up = stim[:, :self._rnn_params.n_bottom_up]
                top_down = stim[:, self._rnn_params.n_bottom_up:]

                p, h, m = self.model([bottom_up, top_down, h, m], training=True)
                policy.append(p)
                activity.append(h)
            policy = tf.stack(policy, axis=1)
            activity = tf.stack(activity, axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels, policy)
            loss = tf.reduce_mean(mask * loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)

        self.opt.learning_rate.assign(learning_rate)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, activity, policy
