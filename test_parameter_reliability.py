import tensorflow as tf
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os
import copy
import layers
import analysis
from actor import ActorSL
from TaskManager import TaskManager, default_tasks
import yaml
import time
import uuid
import glob

gpu_idx = 0
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
"""
tf.config.experimental.set_virtual_device_configuration(
    gpus[gpu_idx],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
"""

class Agent:
    def __init__(self, args, rnn_params):

        self._args = args
        self._rnn_params = rnn_params

        tasks = default_tasks()
        self.n_tasks = len(tasks)
        stim = TaskManager(tasks, batch_size=args.batch_size, tf2=False)

        rnn_params = define_dependent_params(rnn_params, stim)
        self.actor = ActorSL(args, rnn_params, learning_type='supervised')

        self.training_batches = [stim.generate_batch(args.batch_size, to_exclude=[]) for _ in range(args.n_stim_batches)]
        self.dms_batch = stim.generate_batch(args.batch_size, rule=0, include_test=True)
        self.sample_decode_time = [(300+200+300+500)//rnn_params.dt, (300+200+300+980)//rnn_params.dt]

        print('Trainable variables...')
        for v in self.actor.model.trainable_variables:
            print(v.name, v.shape)

        print('Non-trainable variables...')
        for v in self.actor.model.non_trainable_variables:
            print(v.name, v.shape)

        print(self.actor.model.summary())


    def train(self, rnn_params, n_networks, save_fn, original_task_accuracy=None):

        # Set up records in advance, so each results file contains
        # the results for all networks with the same parameter set
        sample_decoding = []
        save_fn = os.path.join(self._args.save_path, save_fn)

        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'original_task_accuracy': original_task_accuracy,
            'sample_decode_time': self.sample_decode_time,
            'loss'                 : [],
            'task_accuracy'        : [],
            'sample_decoding'      : [],
            'final_mean_h'         : [],
            'final_monkey_DMS_data': [],
            'successes'            : []
        }

        for k in range(n_networks):

            # Generate the network weights
            self.actor.RNN.generate_new_weights(rnn_params)
            print('Determing steady-state values...')
            h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0][0])
            print(f"Steady-state activity {np.mean(h_init):2.4f}")

            # For successful-looking networks: record initial sample decode pre-training,
            # then train, and record accuracy/performance
            h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
            results['sample_decoding'].append(analysis.decode_signal(
                                h.numpy(),
                                np.int32(self.dms_batch[4]),
                                self.sample_decode_time))
            sd = results['sample_decoding'][-1]
            print(f"Decoding accuracy {sd[0]:1.3f}, {sd[1]:1.3f}")

            # Train the network
            print('Starting main training loop...')
            tr_acc, tr_loss = np.zeros((self._args.n_iterations, self.n_tasks)), np.zeros(self._args.n_iterations)
            for j in range(self._args.n_iterations):
                t0 = time.time()

                batch = self.training_batches[j%self._args.n_stim_batches]
                learning_rate = np.minimum(
                    self._args.learning_rate,
                    j / self._args.n_learning_rate_ramp * self._args.learning_rate)

                loss, h, policy = self.actor.train(batch, copy.copy(h_init), copy.copy(m_init), learning_rate)

                accuracies = analysis.accuracy_SL_all_tasks(
                                        np.float32(policy),
                                        np.float32(batch[1]),
                                        np.float32(batch[2]),
                                        np.int32(batch[5]),
                                        np.arange(self.n_tasks))

                print(f'Iteration {j} Loss {loss:1.4f} Accuracy {np.mean(accuracies):1.3f} Mean activity {np.mean(h):2.4f} Time {time.time()-t0:2.2f}')
                tr_acc[j, :] = accuracies
                tr_loss[j]   = loss.numpy()

            # Add the necessary elements to the results that we record
            results['loss'].append(tr_loss)
            results['task_accuracy'].append(tr_acc)
            h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
            results['final_mean_h'].append(np.mean(h.numpy(), axis = (0,2)))
            if self._args.save_frs_by_condition:
                results['final_monkey_DMS_data'].append(analysis.average_frs_by_condition(h.numpy(),
                    self.monkey_dms_batch[-3], self.monkey_dms_batch[-1]))
            results['successes'].append(k)

            # Reset the optimizer to facilitate training of the next model
            self.actor.reset_optimizer()

        pickle.dump(results, open(save_fn, 'wb'))

class ParameterReliabilityAssessor:
    def __init__(self, args, search_results=False):
        self._args = args
        self.search_results = search_results
        if search_results:
            self.params_files = glob.glob(f"{args.params_folder}*.pkl")
        else:
            self.params_files = glob.glob(f"{args.params_folder}*.yaml")

    def assess_reliability(self, n_networks):
        for params_fn in self.params_files:

            # check if file already exists in save directory
            save_fn = os.path.join(self._args.save_path, os.path.basename(params_fn))

            if os.path.isfile(save_fn):
                print('File already exists. Skipping.')
                continue

            rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
            rnn_params = argparse.Namespace(**rnn_params)

            if self.search_results:
                results = pickle.load(open(params_fn, 'rb'))
                if len(results['task_accuracy']) == 0:
                    print('Aborting')
                    continue
                accuracy = np.stack(results['task_accuracy'])
                #print(accuracy.shape)
                mean_accuracy = np.mean(accuracy[-10:, :])
                print(f'Mean accuracy {mean_accuracy}')
                if mean_accuracy < 0.8:
                    print('Aborting')
                    continue
                p = vars(results['rnn_params'])
            else:
                p = yaml.load(open(params_fn), Loader=yaml.FullLoader)
            for k, v in p.items():
                if hasattr(rnn_params, k):
                    setattr(rnn_params, k, v)
            agent = Agent(self._args, rnn_params)
            agent.train(
                rnn_params,
                n_networks,
                os.path.basename(params_fn),
                original_task_accuracy=mean_accuracy)

def define_dependent_params(params, stim):

    params.n_input     = stim.n_input
    params.n_actions   = stim.n_output
    params.n_hidden    = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned +  stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned
    params.n_motion_tuned = stim.n_motion_tuned
    params.n_top_down  = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned

    return params



parser = argparse.ArgumentParser('')
parser.add_argument('--n_iterations', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_stim_batches', type=int, default=25)
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--adam_epsilon', type=float, default=1e-7)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/good_params.yaml')
parser.add_argument('--params_folder', type=str, default='./results/run_073121_5tasks/')
parser.add_argument('--training_type', type=str, default='supervised')
parser.add_argument('--save_path', type=str, default='results/test')
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--max_h_for_output', type=float, default=999.)
parser.add_argument('--steady_state_start', type=float, default=1300)
parser.add_argument('--steady_state_end', type=float, default=1700)

args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

################################################################################
# Make ParameterReliabilityAssessor() object and test all param sets in folder
################################################################################
pa = ParameterReliabilityAssessor(args, search_results=True)
pa.assess_reliability(5)
