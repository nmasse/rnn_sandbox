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
from actor import ActorSL
from TaskManager import TaskManager, default_tasks
import yaml
import time
import uuid


gpu_idx = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[gpu_idx],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4900)])

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
        self.actor = ActorSL(args, rnn_params, learning_type='supervised')

        #self.training_batches = [stim.generate_batch(args.batch_size, to_exclude=[self.n_tasks - 1]) for _ in range(args.n_stim_batches)]
        self.training_batches = [stim.generate_batch(args.batch_size, to_exclude=[]) for _ in range(args.n_stim_batches)]
        #self.monkey_dms_batch = stim.generate_batch(args.batch_size, rule=self.n_tasks - 1, include_test=True)
        self.dms_batch = stim.generate_batch(args.batch_size, rule=0, include_test=True)
        self.sample_decode_time = [(300+200+300+500)//rnn_params.dt, (300+200+300+980)//rnn_params.dt]

        print('Trainable variables...')
        for v in self.actor.model.trainable_variables:
            print(v.name, v.shape)

        print('Non-trainable variables...')
        for v in self.actor.model.non_trainable_variables:
            print(v.name, v.shape)

        print(self.actor.model.summary())


    def train(self, rnn_params, counter):

        save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'loss': [],
            'task_accuracy': [],
            'sample_decode_time': self.sample_decode_time,
            'counter': counter,
        }

        self.actor.RNN.generate_new_weights(rnn_params)

        print('Determing steady-state values...')
        h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0][0])
        results['steady_state_h'] = np.mean(h_init)
        print(f"Steady-state activity {results['steady_state_h']:2.4f}")

        if results['steady_state_h'] < 0.01 or results['steady_state_h'] > 1:
            pickle.dump(results, open(save_fn, 'wb'))
            print('Aborting...')
            return False

        print('Determing initial sample decoding accuracy...')
        h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
        results['sample_decoding'] = analysis.decode_signal(
                            h.numpy(),
                            np.int32(self.dms_batch[4]),
                            self.sample_decode_time)
        sd = results['sample_decoding']
        print(f"Decoding accuracy {sd[0]:1.3f}, {sd[1]:1.3f}")

        print('Calculating average spike rates...')
        results['initial_mean_h'] = np.mean(h.numpy(), axis = (0,2))
        if self._args.save_frs_by_condition:
            results['initial_monkey_DMS_data'] = analysis.average_frs_by_condition(h.numpy(),
                self.dms_batch[-3], self.dms_batch[-1])

        print('Starting main training loop...')
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
            results['task_accuracy'].append(accuracies)
            results['loss'].append(loss.numpy())

        h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
        results['final_mean_h'] = np.mean(h.numpy(), axis = (0,2))
        if self._args.save_frs_by_condition:
            results['final_monkey_DMS_data'] = analysis.average_frs_by_condition(h.numpy(),
                self.monkey_dms_batch[-3], self.monkey_dms_batch[-1])

        pickle.dump(results, open(save_fn, 'wb'))
        self.actor.reset_optimizer()
        return True



    def main_loop(self):

        full_runs = 0
        for i in range(1000000):
            print(f'Main loop iteration {i} - Full runs {full_runs}')
            params =  {k:v for k,v in vars(self._rnn_params).items()}
            for k, v in param_ranges.items():
                new_value = np.random.uniform(v[0], v[1] + 1e-16)
                params[k] = new_value

            success = self.train(argparse.Namespace(**params), i)
            if success:
                full_runs += 1


def define_dependent_params(params, stim):

    params.n_input   = stim.n_input
    params.n_actions = stim.n_output
    params.n_hidden = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned
    params.n_top_down = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned

    return params


parser = argparse.ArgumentParser('')
parser.add_argument('--n_iterations', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_stim_batches', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--n_learning_rate_ramp', type=int, default=10)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--lmbda', type=float, default=0.0)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_mod.yaml')
parser.add_argument('--params_range_fn', type=str, default='./rnn_params/param_ranges.yaml')
parser.add_argument('--save_path', type=str, default='./results/test')


args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
param_ranges = yaml.load(open(args.params_range_fn), Loader=yaml.FullLoader)

rnn_params = argparse.Namespace(**rnn_params)
agent = Agent(args, rnn_params, param_ranges)
agent.main_loop()
