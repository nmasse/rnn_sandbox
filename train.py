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
from actor import Actor
from TaskManager import TaskManager, default_tasks, monkey_dms_task
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

        self.training_batches = [stim.generate_batch(args.batch_size, to_exclude=[self.n_tasks - 1]) for _ in range(args.n_stim_batches)]
        self.dms_batch = stim.generate_batch(args.batch_size, rule=0)
        self.monkey_dms_batch = stim.generate_batch(args.batch_size, rule=self.n_tasks - 1)


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

        save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'loss': [],
            'task_accuracy': [],
        }


        self.actor.RNN.generate_new_weights(rnn_params)

        print('Determing steady-state values...')
        h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0])
        results['steady_state_h'] = np.mean(h_init)
        print(f"Steady-state activity {results['steady_state_h']:2.4f}")

        if results['steady_state_h'] < 0.01 or results['steady_state_h'] > 1:
            pickle.dump(results, open(save_fn, 'wb'))
            print('Aborting...')
            return False

        print('Determing initial sample decoding accuracy...')
        h = self.actor.run_batch(self.dms_batch, h_init, m_init)
        results['sample_decode_time'] = [(200+300+600)//rnn_params.dt, (200+300+980)//rnn_params.dt]
        results['sample_decoding'] = analysis.decode_signal(
                            np.float32(h),
                            np.int32(self.dms_batch[4][:,0]),
                            results['sample_decode_time'])
        results['initial_mean_h'] = np.mean(h.numpy(), axis = (0,2))

        sd = results['sample_decoding']
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
            results['task_accuracy'].append(accuracies)
            results['loss'].append(loss.numpy())
            results['final_mean_h'] = np.mean(h.numpy(), axis = (0,2))

            # Generate activity on monkeyDMS batch
            h = self.actor.run_batch(self.monkey_dms_batch[:-1], h_init, m_init)
            results['monkey_DMS_data'] = analysis.average_frs_by_condition(h, 
                self.monkey_dms_batch[-3][:,0], self.monkey_dms_batch[-1][:,0])

        pickle.dump(results, open(save_fn, 'wb'))
        self.actor.reset_optimizer()
        return True



    def main_loop(self):

        full_runs = 0
        for i in range(1000000):
            print(f'Main loop iteration {i} - Full runs {full_runs}')
            #print('Randomly selecting new network parameters...')
            params =  {k:v for k,v in vars(self._rnn_params).items()}
            for k, v in param_ranges.items():
                new_value = np.random.uniform(v[0], v[1])
                #print(k, v, new_value)
                params[k] = new_value

            success = self.train(argparse.Namespace(**params))
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
parser.add_argument('--n_iterations', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_stim_batches', type=int, default=25)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--n_learning_rate_ramp', type=int, default=10)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_mod.yaml')
parser.add_argument('--params_range_fn', type=str, default='./rnn_params/param_ranges_no_normalization.yaml')
parser.add_argument('--save_path', type=str, default='results_no_normalization')

args = parser.parse_args()

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
param_ranges = yaml.load(open(args.params_range_fn), Loader=yaml.FullLoader)

rnn_params = argparse.Namespace(**rnn_params)
agent = Agent(args, rnn_params, param_ranges)
agent.main_loop()
