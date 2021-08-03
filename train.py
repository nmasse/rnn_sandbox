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
from datetime import date
today = date.today()
tf.keras.backend.set_floatx('float32')

def convert(argument):
    return list(map(int, argument.split(',')))

parser = argparse.ArgumentParser('')
parser.add_argument('gpu_idx', type=int)
parser.add_argument('--n_iterations', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_stim_batches', type=int, default=250)
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--adam_epsilon', type=float, default=1e-7)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--max_h_for_output', type=float, default=999.)
parser.add_argument('--steady_state_start', type=float, default=1300)
parser.add_argument('--steady_state_end', type=float, default=1700)
parser.add_argument('--training_type', type=str, default='supervised')
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/good_params.yaml')
parser.add_argument('--params_range_fn', type=str, default='./rnn_params/param_ranges.yaml')
parser.add_argument('--save_path', type=str, default=f'./results/run_{today.strftime("%b-%d-%Y")}/')
parser.add_argument('--ablation_mode', type=str, default=None)
parser.add_argument('--size_range', type=convert, default=[3000])

args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[args.gpu_idx], True)
tf.config.experimental.set_visible_devices(gpus[args.gpu_idx], 'GPU')


class Agent:
    def __init__(self, args, rnn_params, param_ranges, sz=3000):

        self._args = args
        self._rnn_params = rnn_params
        self._param_ranges = param_ranges

        self.tasks = default_tasks()#[:5]
        self.n_tasks = len(self.tasks)
        print(f"Training on {self.n_tasks} tasks")

        # Define E/I number based on argument
        self._rnn_params.n_exc = int(0.8 * sz)
        self._rnn_params.n_inh = int(0.2 * sz)
        self.sz = sz

        alpha = rnn_params.dt / rnn_params.tc_soma
        noise_std = np.sqrt(2/alpha) * rnn_params.noise_input_sd
        stim = TaskManager(
            self.tasks,
            n_motion_tuned=self._rnn_params.n_motion_tuned,
            n_fix_tuned=self._rnn_params.n_fix_tuned,
            batch_size=args.batch_size, input_noise = noise_std, tf2=False)

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


    def train(self, rnn_params, counter):

        save_fn = os.path.join(self._args.save_path, f"{self.sz}_hidden/", 'results_'+str(uuid.uuid4())+'.pkl')

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

        h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
        if np.mean(h) > 10.: # just make sure it's not exploding
            pickle.dump(results, open(save_fn, 'wb'))
            print('Aborting...')
            return False

        print('Determing initial sample decoding accuracy...')
        results['sample_decoding'] = analysis.decode_signal(
                            h.numpy(),
                            np.int32(self.dms_batch[4]),
                            self.sample_decode_time)
        sd = results['sample_decoding']
        print(f"Decoding accuracy {sd[0]:1.3f}, {sd[1]:1.3f}")
        if sd[0] < 0.8: 
            return False


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
            if (j+1)%10==0:
                print("Task accuracies " + " | ".join([f"{accuracies[i]:1.3f}" for i in range(self.n_tasks)]))
            results['task_accuracy'].append(accuracies)
            results['loss'].append(loss.numpy())

        h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
        results['final_mean_h'] = np.mean(h.numpy(), axis = (0,2))
        if self._args.save_frs_by_condition:
            results['final_monkey_DMS_data'] = analysis.average_frs_by_condition(h.numpy(),
                self.monkey_dms_batch[-3], self.monkey_dms_batch[-1])

        results['tasks'] = self.tasks
        pickle.dump(results, open(save_fn, 'wb'))
        self.actor.reset_optimizer()
        return True



    def main_loop(self):

        full_runs = 0
        for i in range(100000000):
            print(f'Main loop iteration {i} - Full runs {full_runs}')
            params =  {k:v for k,v in vars(self._rnn_params).items()}
            for k, v in param_ranges.items():
                if v[0] == v[1]:
                    new_value = v[0]
                else:
                    new_value = np.random.uniform(v[0], v[1])
                params[k] = new_value

            # For all parameters being ablated -- ablate
            if self._args.ablation_mode is not None:
                for k, v in vars(self._rnn_params).items():
                    if self._args.ablation_mode in k and not k.startswith("tc"):
                        params[k] = 0.

            success = self.train(argparse.Namespace(**params), i)
            if success:
                full_runs += 1


def define_dependent_params(params, stim):

    params.n_actions = stim.n_output
    params.n_hidden = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned  + stim.n_fix_tuned # + stim.n_cue_tuned
    params.n_motion_tuned = stim.n_motion_tuned
    params.n_cue_tuned = stim.n_cue_tuned
    params.n_fix_tuned = stim.n_fix_tuned
    params.n_top_down = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned
    params.max_h_for_output = args.max_h_for_output
    print(params)
    return params

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
param_ranges = yaml.load(open(args.params_range_fn), Loader=yaml.FullLoader)

rnn_params = argparse.Namespace(**rnn_params)

# Loop through range of sizes
for sz in args.size_range:
    agent = Agent(args, rnn_params, param_ranges, sz=sz)
    agent.main_loop()
