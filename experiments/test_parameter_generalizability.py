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
from ..tasks.TaskManager import TaskManager, default_tasks 
import yaml
import time
import uuid
import glob

def convert(argument):
    return list(map(int, argument.split(',')))

parser = argparse.ArgumentParser('')
parser.add_argument('gpu_idx', type=int)
parser.add_argument('fold', type=int, help='Subset # of params to test')
parser.add_argument('--n_training_iterations', type=int, default=175)
parser.add_argument('--n_evaluation_iterations', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_stim_batches', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--adam_epsilon', type=float, default=1e-7)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--top_down_grad_multiplier', type=float, default=0.1)
parser.add_argument('--clip_grad_norm', type=float, default=1.)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/good_params.yaml')
parser.add_argument('--params_folder', type=str, default='./rnn_params/generalizability_test/')
parser.add_argument('--training_type', type=str, default='supervised')
parser.add_argument('--save_path', type=str, default='results/generalizability_assessment')
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--max_h_for_output', type=float, default=5.)
parser.add_argument('--steady_state_start', type=float, default=1300)
parser.add_argument('--steady_state_end', type=float, default=1700)
parser.add_argument('--task_set', type=str, default='challenge')
parser.add_argument('--model_type', type=str, default='model_experimental')
parser.add_argument('--restrict_output_to_exc', type=bool, default=False)
parser.add_argument('--size_range', type=convert, default=[2500])

args = parser.parse_args()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[args.gpu_idx], 'GPU')

class Agent:
    def __init__(self, args, rnn_params, sz=2500):

        self._args = args
        self._rnn_params = rnn_params

        tasks = default_tasks(args.task_set)
        self.n_tasks = len(tasks)
        self.tasks = tasks

        # Define E/I number based on argument
        self._rnn_params.n_exc = int(0.8 * sz)
        self._rnn_params.n_inh = int(0.2 * sz)
        self._rnn_params.restrict_output_to_exc = args.restrict_output_to_exc
        self.sz = sz
        self._rnn_params.n_motion_tuned = 32

        alpha = rnn_params.dt / rnn_params.tc_soma
        noise_std = np.sqrt(2/alpha) * rnn_params.noise_input_sd
        stim = TaskManager(
            self.tasks,
            n_motion_tuned=self._rnn_params.n_motion_tuned,
            n_fix_tuned=self._rnn_params.n_fix_tuned,
            batch_size=args.batch_size, input_noise = noise_std, tf2=False)

        rnn_params = define_dependent_params(self._rnn_params, stim)
        self.actor = ActorSL(self._args, self._rnn_params, learning_type='supervised')

        self.training_batches = [stim.generate_batch(args.batch_size, to_exclude=[0]) for _ in range(args.n_stim_batches)]
        self.dms_batch = stim.generate_batch(args.batch_size, rule=0, include_test=True)
        self.sample_decode_time = [(300+200+300+500)//rnn_params.dt, (300+200+300+980)//rnn_params.dt]
        self.continue_resets = [1800//rnn_params.dt, 2600//rnn_params.dt]

        print('Trainable variables...')
        for v in self.actor.model.trainable_variables:
            print(v.name, v.shape)

        print('Non-trainable variables...')
        for v in self.actor.model.non_trainable_variables:
            print(v.name, v.shape)

        print(self.actor.model.summary())
        self._args.n_iterations = self._args.n_training_iterations + self._args.n_evaluation_iterations


    def train(self, rnn_params, n_networks, save_fn, original_task_accuracy=None):

        # Set up records in advance, so each results file contains
        # the results for all networks with the same parameter set
        sample_decoding = []
        save_fn = os.path.join(self._args.save_path, os.path.splitext(save_fn)[0] + ".pkl")

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
            'successes'            : [],
            'initial_mean_h'       : [],
            'steady_state_h'       : []
        }
        

        for k in range(n_networks):

            # Generate the network weights
            self.actor.RNN.generate_new_weights(rnn_params)
            print('Determing steady-state values...')
            h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0][0])
            print(f"Steady-state activity {np.mean(h_init):2.4f}")
            results['steady_state_h'].append(np.mean(h_init))

            # For successful-looking networks: record initial sample decode pre-training,
            # then train, and record accuracy/performance
            h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))

            results['initial_mean_h'].append(np.mean(h.numpy(), axis = (0,2)))
            results['sample_decoding'].append(analysis.decode_signal(
                                h.numpy(),
                                np.int32(self.dms_batch[4]),
                                self.sample_decode_time))
            sd = results['sample_decoding'][-1]
            print(f"Decoding accuracy {sd[0]:1.3f}, {sd[1]:1.3f}")

            # Train the network
            print('Starting main training loop...')
            tr_acc, tr_loss = np.zeros((self._args.n_iterations, self.n_tasks - 1)), np.zeros(self._args.n_iterations)
            for j in range(self._args.n_iterations):
                t0 = time.time()

                batch = self.training_batches[j%self._args.n_stim_batches]
                if j >= self._args.n_training_iterations:
                    learning_rate = 0.
                else:
                    learning_rate = np.minimum(
                        self._args.learning_rate,
                        (j+1) / self._args.n_learning_rate_ramp * self._args.learning_rate)

                loss, h, policy = self.actor.train(batch, copy.copy(h_init), copy.copy(m_init), learning_rate)

                accuracies = analysis.accuracy_SL_all_tasks(
                                        np.float32(policy),
                                        np.float32(batch[1]),
                                        np.float32(batch[2]),
                                        np.int32(batch[5]),
                                        np.arange(1, self.n_tasks),
                                        continue_resets=self.continue_resets)

                tr_acc[j, :] = accuracies
                tr_loss[j]   = loss.numpy()
                print(f'Iteration {j} Loss {loss:1.4f} Accuracy {np.mean(accuracies):1.3f} Mean activity {np.mean(h):2.4f} Time {time.time()-t0:2.2f}')
                if (j+1)%10==0:
                    t = np.maximum(0, j-25)
                    acc = np.stack(tr_acc[:j],axis=0)
                    acc = np.mean(acc[t:, :], axis=0)
                    print("Task accuracies " + " | ".join([f"{acc[i]:1.3f}" for i in range(self.n_tasks - 1)]))

                if (j+1) % 100 == 0:
                    results[f'iter_{j + 1}_mean_h'] = np.mean(h.numpy(), axis = (0,2))

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

        pickle.dump(results, open(save_fn, 'wb'))

class ParameterGeneralizabilityAssessor:
    def __init__(self, args):
        self._args = args
        self.params_files = glob.glob(f"{args.params_folder}*.yaml")

        # Additional filter: only do this for files that have the correct 
        # fold number
        self.params_files = [fn for fn in self.params_files if f"fold={self._args.fold}_" in fn] 

    def assess_generalizability(self, n_networks):

        for params_fn in self.params_files:

            # Check if file already exists in save directory
            save_fn = os.path.join(self._args.save_path, os.path.basename(params_fn))
            if os.path.exists(save_fn[:-5] + ".pkl"):
                continue
            pickle.dump({}, open(save_fn[:-5]+".pkl", 'wb'))

            mean_accuracy = os.path.basename(params_fn)
            start = mean_accuracy.find("acc=")+4
            end = start + 6
            try:
                mean_accuracy = float(mean_accuracy[start:end])
            except ValueError:
                mean_accuracy = None


            rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
            rnn_params = argparse.Namespace(**rnn_params)

            p = yaml.load(open(params_fn), Loader=yaml.FullLoader)
            for k, v in p.items():
                if hasattr(rnn_params, k):
                    setattr(rnn_params, k, v)
            agent = Agent(self._args, rnn_params, sz=args.size_range[0])
            agent.train(
                rnn_params,
                n_networks,
                os.path.basename(params_fn),
                original_task_accuracy=mean_accuracy)

def define_dependent_params(params, stim):

    params.n_actions = stim.n_output
    params.n_hidden = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned  + stim.n_fix_tuned + stim.n_cue_tuned
    params.n_motion_tuned = stim.n_motion_tuned
    params.n_cue_tuned = stim.n_cue_tuned
    params.n_fix_tuned = stim.n_fix_tuned
    params.n_top_down = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned
    params.max_h_for_output = args.max_h_for_output

    return params


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

################################################################################
# Make ParameterGeneralizabilityAssessor() object and test all param sets in folder
################################################################################
pa = ParameterGeneralizabilityAssessor(args)
pa.assess_generalizability(5)
