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
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor:
    def __init__(self, args, rnn_params, learning_type='supervised'):
        self._args = args
        self._rnn_params = rnn_params
        self.learning_type = learning_type
        #self.n_hidden = rnn_params.n_exc + rnn_params.n_inh + rnn_params.n_mod
        self.n_hidden = rnn_params.n_exc + rnn_params.n_inh
        self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-07)


    def create_model(self):

        #init = tf.keras.initializers.Orthogonal(gain=0.1)
        #init = tf.keras.initializers.GlorotNormal()
        #init = 0.1 * init(shape=(self.n_hidden, self._rnn_params.n_actions))

        x_input = tf.keras.Input((self._rnn_params.n_input,)) # Stimulus input
        s_input = tf.keras.Input((self.n_hidden,)) # Soma
        m_input = tf.keras.Input((self.n_hidden,)) # Modulator

        self.RNN = layers.Model(self._rnn_params)

        soma, modulator = self.RNN.model((x_input, s_input, m_input))
        h = tf.nn.relu(soma)
        policy = Dense(self._rnn_params.n_actions, activation='linear', name='policy')(h)
        self.model = tf.keras.models.Model([x_input, s_input, m_input], [policy, soma, modulator])



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


    def train(self, batch):

        stimulus = batch[0]
        labels = batch[1]
        mask = batch[2]

        s, m = self.RNN.intial_activity(self._args.batch_size)

        with tf.GradientTape(persistent=False) as tape: #not sure persistent matters

            policy = []
            activity = []
            for stim in tf.unstack(stimulus,axis=1):
                p, s, m = self.model([stim, s, m], training=True)
                policy.append(p)
                activity.append(tf.nn.relu(s))
            policy = tf.stack(policy,axis=1)
            activity = tf.stack(activity,axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels, policy)
            loss = tf.reduce_mean(mask * loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        grads_and_vars = []
        m = np.ones((41,1),dtype=np.float32)
        m[:33,0] = 0.
        m = tf.constant(m)
        for g,v in zip(grads, self.model.trainable_variables):

            if self._args.train_context_only:
                if not ('input' in v.name or 'policy' in v.name):
                    g *= 0.
                if 'input' in v.name:
                    g *= m


            grads_and_vars.append((g,v))

        self.opt.apply_gradients(grads_and_vars)

        return loss, activity, policy


class Agent:
    def __init__(self, args, rnn_params, stim=None):

        self._args = args
        self._rnn_params = rnn_params
        self.actor = Actor(args, rnn_params, learning_type='supervised')
        if stim is None:
            self.stim = CognitiveTasks(rnn_params, batch_size=args.batch_size)
        else:
            self.stim = stim

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



    def train(self):

        results = {
            'epiosde_scores': [],
            'time_steps': [],
            'args': args
            }

        for j, batch in enumerate(self.stim.dataset):
            #plt.imshow(batch[0][0,...],aspect='auto')
            #plt.colorbar()
            #plt.show()
            loss, h, policy = self.actor.train(batch)

            accuracy = analysis.accuracy_SL(policy, np.float32(batch[1]), np.float32(batch[2]))
            print(f'Iteration {j} Loss {loss:1.4f} Accuracy {accuracy:1.3f} Mean activity {np.mean(h):2.4f}')
            if j == 0 or j == 5000:
                plt.plot(np.mean(h,axis=(0,2)),'b')
                plt.plot(np.mean(h[:,:,:self._rnn_params.n_exc],axis=(0,2)),'r')
                plt.plot(np.mean(h[:,:,self._rnn_params.n_exc:],axis=(0,2)),'g')
                plt.show()
                t = np.arange(0,125,5)*20 - 500
                acc = analysis.decode_signal(np.float32(h), np.int32(batch[4]), list(range(0,125,5)))
                plt.plot(t, acc)
                plt.show()

            if j > self._args.n_iterations:
                break



parser = argparse.ArgumentParser('')
parser.add_argument('--n_iterations', type=int, default=6000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.002)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn.yaml')
parser.add_argument('--test_stim', type=bool, default=True)
parser.add_argument('--train_context_only', type=bool, default=True)

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

if args.test_stim:
    tasks = default_tasks()
    #tasks = tasks[0:1]
    #print(tasks)
    #1/0
    stim = TaskManager(tasks, batch_size=args.batch_size, tf2=True)

    # Adjust n input units / n outputs (actions) as per stim
    rnn_params.n_input   = stim.n_input
    rnn_params.n_actions = stim.n_output
else:
    stim = None
agent = Agent(args, rnn_params, stim=stim)
agent.train()
