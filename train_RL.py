import tensorflow as tf
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os
import copy
import matplotlib.pyplot as plt
import layers
import analysis
from actor import ActorRL
from TaskManager import TaskManager, TaskGym, default_tasks
import yaml
import time
import uuid


gpu_idx = 3
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[gpu_idx],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5100)])

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Agent:
    def __init__(self, args, rnn_params):

        self._args = args
        self._rnn_params = rnn_params

        tasks = default_tasks()
        self.n_tasks = len(tasks)
        self.env = TaskGym(tasks, args.batch_size, buffer_size=1000, new_task_prob=1.)

        rnn_params = define_dependent_params(rnn_params, self.env.task_manager)
        self.actor = ActorRL(args, rnn_params)

        print('Trainable variables...')
        for v in self.actor.model.trainable_variables:
            print(v.name, v.shape)

        print(self.actor.model.summary())

    def gae_target(self, rewards, values, last_value, done):

        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        nsteps = rewards.shape[0]
        for k in reversed(range(nsteps)):
            if k == nsteps - 1:
                nextnonterminal = 1.0 - done[-1,:]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - done[k+1,:]
                nextvalues = values[k+1,:]

            delta = rewards[k, :] + self._args.gamma * nextvalues * nextnonterminal - values[k, :]
            gae_cumulative = self._args.gamma * self._args.lmbda * gae_cumulative * nextnonterminal + delta
            gae[k,:] = gae_cumulative
        n_step_targets = gae + values

        return gae, n_step_targets


    def train(self):

        save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'loss': [],
            'task_accuracy': []
        }


        self.actor.RNN.generate_new_weights(self._rnn_params)

        time_steps =  np.zeros((self._args.batch_size), dtype=np.int32)
        n_completed_episodes =  0
        t0 = time.time()
        episode_reward = np.zeros((self._args.batch_size), dtype=np.float32)
        current_episode_time = np.zeros((self._args.batch_size), dtype=np.int32)

        state = self.env.reset_all()

        h = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype = np.float32)
        m = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype = np.float32)

        print(h.shape, m.shape)
        for ep in range(self._args.n_episodes):

            alpha = np.clip(ep / self._args.action_std_episode_anneal, 0., 1.)
            actor_continuous_std = (1-alpha) * self._args.start_action_std +  alpha * self._args.end_action_std
            actor_continuous_std = np.float32(actor_continuous_std)

            states, actions, values , rewards, old_policies, cont_actions = [], [], [], [], [], []
            dones = [np.zeros((self._args.batch_size), dtype=np.float32)]

            for t in range(self._args.time_horizon):
                print('Time', t)
                time_steps += 1
                current_episode_time += 1
                h, m, log_policy, action, value = self.actor.get_actions([state, h, m])
                print('pol', log_policy, np.mean(h))
                next_state, reward, done = self.env.step_all(action)
                episode_reward += reward

                states.append(copy.copy(state))
                old_policies.append(np.float32(log_policy))
                values.append(np.squeeze(value))
                actions.append(action)
                rewards.append(reward)
                dones.append(np.float32(done))
                #cont_actions.append(cont_action)
                state = next_state


            _, _, _, _, next_values = self.actor.get_actions([state, h, m])

            gaes, td_targets = self.gae_target(
                np.stack(rewards, axis = 0),
                np.stack(values, axis = 0),
                np.squeeze(next_values),
                np.stack(dones, axis = 0))

        






def define_dependent_params(params, stim):

    params.n_input   = stim.n_input
    params.n_actions = stim.n_output
    params.n_hidden = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned
    params.n_top_down = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned

    return params




parser = argparse.ArgumentParser('')
parser.add_argument('--n_episodes', type=int, default=10000)
parser.add_argument('--time_horizon', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--n_stim_batches', type=int, default=250)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--n_learning_rate_ramp', type=int, default=10)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--training_type', type=str, default='supervised')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_small.yaml')
parser.add_argument('--save_path', type=str, default='./results/test')
parser.add_argument('--start_action_std', type=float, default=0.1)
parser.add_argument('--end_action_std', type=float, default=0.01)
parser.add_argument('--action_std_episode_anneal', type=float, default=5000)

args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)

rnn_params = argparse.Namespace(**rnn_params)
agent = Agent(args, rnn_params)
agent.train()
