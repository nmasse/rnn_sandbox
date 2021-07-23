import tensorflow as tf
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os
import copy
import layers
import analysis
from actor import Actor
from TaskManager import TaskManager, default_tasks
import yaml
import time
import uuid

gpu_idx = 0
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[gpu_idx],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Agent:
    def __init__(self, args, rnn_params, param_ranges):

        self._args = args
        self._rnn_params = rnn_params
        self._param_ranges = param_ranges

        tasks = default_tasks()
        self.n_tasks = len(tasks)
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



    def train(self, rnn_params):

        attempts = 0
        while True:
            attempts += 1
            self.actor.RNN.generate_new_weights(rnn_params)
            print('Determing steady-state values...')
            h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0])
            print(f"Steady-state activity {np.mean(h_init):2.4f}")

            if np.mean(h_init) < 0.01 or np.mean(h_init) > 1:
                print('Resmapling weights...')

            if attempts > 100:
                print('Ending. Too many attempts.')
                return


        print('Determing initial sample decoding accuracy...')
        h = self.actor.run_batch(self.dms_batch, h_init, m_init)
        sample_decode_time = [(200+300+600)//rnn_params.dt, (200+300+980)//rnn_params.dt]
        sample_decoding = analysis.decode_signal(
                            np.float32(h),
                            np.int32(self.dms_batch[4]),
                            sample_decode_time)


        sd = sample_decoding
        print(f"Decoding accuracy {sd[0]:1.3f}, {sd[1]:1.3f}")

        print('Starting main training loop...')
        for j in range(self._args.n_iterations):
            t0 = time.time()

            batch = self.training_batches[j%self._args.n_stim_batches]
            learning_rate = np.minimum(
                self._args.learning_rate,
                j / self._args.n_learning_rate_ramp * self._args.learning_rate)

            loss, h, policy = self.actor.train(batch, h_init, m_init, learning_rate)

            accuracies = analysis.accuracy_all_tasks(
                                    np.float32(policy),
                                    np.float32(batch[1]),
                                    np.float32(batch[2]),
                                    np.int32(batch[5]),
                                    list(range(self.n_tasks)))
            print(f'Iteration {j} Loss {loss:1.4f} Accuracy {np.mean(accuracies):1.3f} Mean activity {np.mean(h):2.4f} Time {time.time()-t0:2.2f}')

        self.actor.model.save(self._args.save_model_path)
        print(f'Model saved to {self._args.save_model_path}')



def define_dependent_params(params, stim):

    params.n_input   = stim.n_input
    params.n_actions = stim.n_output
    params.n_hidden = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned
    params.n_top_down = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned

    return params




parser = argparse.ArgumentParser('')
parser.add_argument('--n_iterations', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_stim_batches', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.002)
parser.add_argument('--n_learning_rate_ramp', type=int, default=10)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_mod.yaml')
parser.add_argument('--save_model_path', type=str, default='saved_models/temp0')

args = parser.parse_args()

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

if args.rnn_params_fn.endswith('.yaml'):
    rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
elif args.rnn_params_fn.endswith('.pkl'):
    rnn_params = pickle.load(open(args.rnn_params_fn,'rb'))['rnn_params']
    rnn_params = argparse.Namespace(**rnn_params)
else:
    assert False, f"{args.rnn_params_fn} is uncrecongized file type"
agent = Agent(args, rnn_params, param_ranges)
agent.train(rnn_params)
