import tensorflow as tf
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import copy
import matplotlib.pyplot as plt
import layers
import analysis
from stimulus import CognitiveTasks
from TaskManager import TaskManager, default_tasks
import yaml
from tensorflow.keras.layers import Dense
import time

gpu_idx = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[gpu_idx],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor:
    def __init__(self, args, rnn_params, learning_type='supervised'):
        self._args = args
        self._rnn_params = rnn_params
        self.learning_type = learning_type
        self.create_model()
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


class Agent:
    def __init__(self, args, rnn_params):

        self._args = args
        self._rnn_params = rnn_params

        tasks = default_tasks()
        stim = TaskManager(tasks, batch_size=args.batch_size, tf2=False)

        rnn_params = define_dependent_params(rnn_params, stim)
        self.actor = Actor(args, rnn_params, learning_type='supervised')

        self.training_batches = [stim.generate_batch(args.batch_size) for _ in range(args.n_stim_batches)]
        self.dms_batch = stim.generate_batch(args.batch_size, rule=0)


        print('Trainable variables...')
        for v in self.actor.model.trainable_variables:
            print(v.name, v.shape)

        print(self.actor.model.summary())


    def gae_target(self, rewards, values, last_value, done):

        gae = np.zeros(rewards.shape, dtype=np.float32)
        gae_cumulative = 0.
        nsteps = rewards.shape[0]
        for k in reversed(range(nsteps)):
            if k == nsteps - 1:
                nextnonterminal = 1.0 - done[-1,:]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - done[k+1,:]
                nextvalues = values[k+1,:]

            delta = rewards[k, :] + args.gamma * nextvalues * nextnonterminal - values[k, :]
            gae_cumulative = args.gamma * args.lmbda * gae_cumulative * nextnonterminal + delta
            gae[k,:] = gae_cumulative
        n_step_targets = gae + values

        return gae, np.float32(n_step_targets)



    def train(self, id):

        results = {
            'args': self._args,
            'rnn_params': self._rnn_params,
            'loss': [],
            'task_accuracy': [],
        }


        self.actor.RNN.generate_new_weights(self._rnn_params)

        print('Determing steady-stae values...')
        h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0])
        results['initial_mean_h'] = np.mean(h_init)

        print('Determing initial sample decoding accuracy...')
        h = self.actor.run_batch(self.dms_batch, h_init, m_init)
        results['sample_decode_time'] = (200+300+980) // self._rnn_params.dt
        results['sample_decoding'] = analysis.decode_signal(
                            np.float32(h),
                            np.int32(self.dms_batch[4]),
                            [results['sample_decode_time']])[0]

        print(f"Decoding accuracy {results['sample_decoding']:1.3f}")

        print('Starting main training loop...')
        for j in range(self._args.n_iterations):
            t0 = time.time()

            batch = self.training_batches[j%self._args.n_stim_batches]
            learning_rate = np.minimum(
                self._args.learning_rate,
                j / self._args.n_learning_rate_ramp * self._args.learning_rate)

            loss, h, policy = self.actor.train(batch, h_init, m_init, learning_rate)

            accuracy = analysis.accuracy_SL(policy, np.float32(batch[1]), np.float32(batch[2]))
            print(f'Iteration {j} Loss {loss:1.4f} Accuracy {accuracy:1.3f} Mean activity {np.mean(h):2.4f} Time {time.time()-t0:2.2f}')
            results['loss'].append(loss.numpy())
            results['task_accuracy'].append(accuracy)


        save_fn = 'results/results'+str(id)+'.pkl'
        pickle.dump(results, open(save_fn, 'wb'))
        print(results)
        self.actor.reset_optimizer()



def define_dependent_params(params, stim):

    params.n_input   = stim.n_input
    params.n_actions = stim.n_output
    params.n_hidden = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned
    params.n_top_down = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned

    return params



parser = argparse.ArgumentParser('')
parser.add_argument('--n_iterations', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_stim_batches', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--n_learning_rate_ramp', type=int, default=10)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_mod.yaml')

args = parser.parse_args()

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
rnn_params = argparse.Namespace(**rnn_params)
print('RNN parameters:')
for k, v in vars(rnn_params).items():
    print(k,':', v)
print()

agent = Agent(args, rnn_params)
agent.train(0)
