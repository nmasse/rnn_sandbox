import tensorflow as tf
import numpy as np
import pickle
import os
import copy
import yaml
import time
import uuid
import glob
import logging
from collections import defaultdict
import h5py
from . import layers
from . import analysis
from .actor import ActorSL
from tasks.TaskManager import TaskManager, default_tasks

tf.keras.backend.set_floatx('float32')
np.set_printoptions(precision=3)

class Agent:
    def __init__(self, args, rnn_params, param_ranges=None, sz=2500):

        # Select GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args.gpu_idx], 'GPU')

        self._args = args
        self._rnn_params = rnn_params
        self._param_ranges = param_ranges

        self.tasks = default_tasks(self._args.task_set)
        self.n_tasks = len(self.tasks)

        # Define E/I number based on argument
        self._rnn_params.n_exc = int(0.8 * sz)
        self._rnn_params.n_inh = int(0.2 * sz)
        self._rnn_params.restrict_output_to_exc = args.restrict_output_to_exc
        self.sz = sz

        # Halve n motion tuned 
        self._rnn_params.n_motion_tuned = self._rnn_params.n_motion_tuned // 2

        # Establish task 
        alpha = rnn_params.dt / rnn_params.tc_soma
        noise_std = np.sqrt(2/alpha) * rnn_params.noise_input_sd
        stim = TaskManager(
            self.tasks,
            n_motion_tuned=self._rnn_params.n_motion_tuned,
            n_fix_tuned=self._rnn_params.n_fix_tuned,
            batch_size=args.batch_size, input_noise = noise_std, tf2=False)

        # Define actor
        self.rnn_params = define_dependent_params(self._rnn_params, stim, args)
        self.actor = ActorSL(args, self.rnn_params, learning_type='supervised',
            n_RFs=2)

        self.training_batches = [stim.generate_batch(args.batch_size, to_exclude=[]) \
            for _ in range(args.n_stim_batches)]
        self.eval_batch = stim.generate_batch(args.batch_size, rule=0, include_test=True)
        self.sample_decode_time = [(300+200+300+500)//rnn_params.dt, # mid delay
            (300+200+300+980)//rnn_params.dt] # late delay

        self._args.n_iterations = self._args.n_training_iterations + \
            self._args.n_evaluation_iterations

        # Set up epoch bounds for dimensionality computation per epoch
        timing = {'dead_time'  : 300,
                  'fix_time'   : 200,
                  'sample_time': 300,
                  'delay_time' : 1000,
                  'test_time'  : 300}
        self.epoch_bounds = {}
        start_time = 0
        for k, v in timing.items():
            cur_rng = range(start_time, start_time + v // self.rnn_params.dt)
            self.epoch_bounds[k[:k.find("_")]] = cur_rng
            start_time += v // self.rnn_params.dt

        self.to_save = np.array([0, 5, 10, 50, 100, 150, 199])
        if self._args.save_activities:

            self.eval_batches = []
            for i in range(self.n_tasks):
                for j in range(2):
                    b = stim.generate_batch(args.batch_size, rule=i, include_test=False)
                    self.eval_batches.append(b)

        print(self.actor.model.summary())
        


    def sweep(self, rnn_params, counter):

        save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        if not os.path.exists(self._args.save_path):
            os.makedirs(self._args.save_path)


        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'loss': [],
            'task_accuracy': [],
            'sample_decode_time': self.sample_decode_time,
            'counter': counter,
        }


        self.actor.RNN.generate_new_weights(rnn_params)


        ########################################################################
        # Determine baseline states
        ########################################################################
        h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0][0])
        results['steady_state_h'] = np.mean(h_init)
        print(f"Steady-state activity {results['steady_state_h']:2.4f}")

        # If steady state activity not in reasonable range, abort
        if results['steady_state_h'] < 0.01 or results['steady_state_h'] > 1.:
            pickle.dump(results, open(save_fn, 'wb'))
            return False

        h, _ = self.actor.forward_pass(self.eval_batch[0], 
            copy.copy(h_init), copy.copy(m_init))

        if np.mean(h) > 10. or not np.isfinite(np.mean(h)): # just make sure it's not exploding
            pickle.dump(results, open(save_fn, 'wb'))
            print('Aborting...')
            return False

        print('Determing initial sample decoding accuracy...')
        results['sample_decoding'] = analysis.decode_signal(
                            h.numpy(),
                            np.int32(self.eval_batch[4]),
                            self.sample_decode_time)
        sd = results['sample_decoding']
        print(f"Decoding accuracy s0: {sd[:,0]} s1: {sd[:,1]}")
        if self._args.aggressive and np.mean(sd[1,:]) < self._args.min_sample_decode:
            pickle.dump(results, open(save_fn, 'wb'))
            print("Aborting...")
            return False

        print('Calculating average spike rates...')
        results['initial_mean_h'] = np.mean(h.numpy(), axis = (0,2))

        print('Calculating dimensionality...')
        results['dimensionality'] = analysis.dimensionality(h.numpy(), self.epoch_bounds.values())
        print(f"Dimensionality per epoch: {results['dimensionality']}")

        h_std = np.std(np.clip(h.numpy(), 0., self._args.max_h_for_output))
        print(f'Spike rate standard deviation {h_std:1.4f}')

        print('Starting main training loop...')
        for j in range(self._args.n_training_iterations+self._args.n_evaluation_iterations):
            t0 = time.time()

            batch = self.training_batches[j%self._args.n_stim_batches]
            if j >= self._args.n_training_iterations:
                learning_rate = 0.
            else:
                learning_rate = np.minimum(
                    self._args.learning_rate,
                    (j+1) / self._args.n_learning_rate_ramp * self._args.learning_rate)


            loss, h, policy = self.actor.train(batch, copy.copy(h_init), 
                copy.copy(m_init), learning_rate)

            accuracies = analysis.accuracy_SL_all_tasks(
                                    np.float32(policy),
                                    np.float32(batch[1]),
                                    np.float32(batch[2]),
                                    np.int32(batch[5]),
                                    np.arange(self.n_tasks))


            print(f'Iteration {j} Loss {loss:1.4f} Accuracy {np.mean(accuracies):1.3f}\
                 Mean activity {np.mean(h):2.4f} Time {time.time()-t0:2.2f}')
            results['task_accuracy'].append(accuracies)
            results['loss'].append(loss.numpy())
            if (j+1)%10==0:
                t = np.maximum(0, j-25)
                acc = np.stack(results['task_accuracy'],axis=0)
                acc = np.mean(acc[t:, :], axis=0)
                print("Task accuracies " + " | ".join([f"{acc[i]:1.3f}" for i in range(self.n_tasks)]))

            if (j+1) % 100 == 0:
                results[f'iter_{j + 1}_mean_h'] = np.mean(h.numpy(), axis = (0,2))

        h, _ = self.actor.forward_pass(self.eval_batch[0], copy.copy(h_init), copy.copy(m_init))
        results['final_mean_h'] = np.mean(h.numpy(), axis = (0,2))

        results['tasks'] = self.tasks
        pickle.dump(results, open(save_fn, 'wb'))
        self.actor.reset_optimizer()
        return True


    def train(self, rnn_params, n_networks, save_fn, original_task_accuracy=None):

        # Set up records in advance, so each results file contains
        # the results for all networks with the same parameter set
        sample_decoding = []
        save_fn = os.path.join(self._args.save_path, os.path.splitext(save_fn)[0])

        if self._args.save_activities:
            self.outfile = h5py.File(save_fn + ".h5", 'a')

        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'original_task_accuracy': original_task_accuracy,
            'sample_decode_time': self.sample_decode_time,
            'loss'                 : [],
            'task_accuracy'        : [],
            'sample_decoding'      : [],
            'final_mean_h'         : [],
            'successes'            : [],
            'initial_mean_h'       : [],
            'steady_state_h'       : [],
            'dimensionality'       : []
        }

        training_results = defaultdict(list)
        

        for k in range(n_networks):

            # Generate the network weights
            self.actor.RNN.generate_new_weights(self._rnn_params, random_seed=k+20)

            print('Determing steady-state values...')
            h_init, m_init, h = self.actor.determine_steady_state(self.training_batches[0][0])
            print(f"Steady-state activity {np.mean(h_init):2.4f}")
            results['steady_state_h'].append(np.mean(h_init))

            h, _ = self.actor.forward_pass(self.eval_batch[0], copy.copy(h_init), copy.copy(m_init))

            results['initial_mean_h'].append(np.mean(h.numpy(), axis = (0,2)))
            results['sample_decoding'].append(analysis.decode_signal(
                                h.numpy(),
                                np.int32(self.eval_batch[4]),
                                self.sample_decode_time))
            sd = results['sample_decoding'][-1]
            print(f"Decoding accuracy {sd[:,0], sd[:,1]}")

            print('Calculating dimensionality...')
            results['dimensionality'].append(analysis.dimensionality(h.numpy(), self.epoch_bounds.values()))
            print(f"Dimensionality per epoch: {results['dimensionality']}")

            # Train the network
            print('Starting main training loop...')
            tr_acc, tr_loss = np.zeros((self._args.n_iterations, self.n_tasks)), np.zeros(self._args.n_iterations)
            
            train_accs = np.zeros((len(self.to_save), self.n_tasks))
            train_weights = defaultdict(list)

            for j in range(self._args.n_iterations):
                t0 = time.time()

                if j in self.to_save and self._args.save_activities:
                    # Do for weights too
                    names = [weight.name for layer in self.actor.model.layers for weight in layer.weights]
                    weights = self.actor.model.get_weights()
                    for n, w in zip(names, weights):
                        train_weights[n].append(w)

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
                                        np.arange(self.n_tasks))
                tr_acc[j, :] = accuracies
                tr_loss[j]   = loss.numpy()
                print(f'Iteration {j} Loss {loss:1.4f} Accuracy {np.mean(accuracies):1.3f} ' + 
                    f'Mean activity {np.mean(h):2.4f} Time {time.time()-t0:2.2f}')
                if (j+1)%10==0:
                    t = np.maximum(0, j-25)
                    acc = np.stack(tr_acc[:j],axis=0)
                    acc = np.mean(acc[t:, :], axis=0)
                    res_str = "Task accuracies " + " | ".join([f"{acc[i]:1.2f}" for i in range(self.n_tasks)])
                    print(res_str)
                    logging.debug(f"\tIter {j+1}: {res_str}")

                # Store activity for every specified iteration
                if j in self.to_save and self._args.save_activities:
                    
                    cur_b = int(np.where(self.to_save == j)[0])
                    train_acts_j = np.zeros((1, h.shape[0]*2*self.n_tasks, h.shape[1], h.shape[2]), dtype=np.float16)

                    for k in range(len(self.eval_batches)):
                        batch = self.eval_batches[k]

                        # Run eval batch
                        loss, h, policy = self.actor.train(batch, copy.copy(h_init), copy.copy(m_init), 0.)

                        accuracies = analysis.accuracy_SL_all_tasks(
                                                np.float32(policy),
                                                np.float32(batch[1]),
                                                np.float32(batch[2]),
                                                np.int32(batch[5]),
                                                np.arange(self.n_tasks))

                        h = np.float32(h.numpy())

                        train_acts_j[0,k*self._args.batch_size:(k+1)*self._args.batch_size,...] = h
                        train_accs[cur_b] += accuracies

                    if j == self.to_save[0]:
                        # Create h5 dataset at first
                        self.outfile.create_dataset('data', data=train_acts_j, compression="gzip", 
                            chunks=True, maxshape=(None,*train_acts_j.shape[1:]))
                    else:
                        # Append new data to it
                        self.outfile['data'].resize((cur_b + 1), axis=0)
                        self.outfile['data'][-1:] = train_acts_j

            # Add the necessary elements to the results that we record
            results['loss'].append(tr_loss)
            results['task_accuracy'].append(tr_acc)
            if self._args.save_activities:
                train_rules = np.array([batch[5] for batch in self.eval_batches]).flatten()
                train_labels = np.array([np.argmax(batch[1][:,-2,:].squeeze(), axis=1) for batch in self.eval_batches]).flatten()
                train_samples = np.vstack([np.int16(batch[4]) for batch in self.eval_batches])

                training_results['train_rules'].append(train_rules)
                training_results['train_accs'].append(train_accs / 2)
                training_results['train_labels'].append(train_labels)
                training_results['train_samples'].append(train_samples)
                training_results['save_iters'] = self.to_save

                names = [weight.name for layer in self.actor.model.layers for weight in layer.weights]
                for n in names:
                    training_results[n] = train_weights[n]

                # If needed: save this out!
                if self._args.do_overwrite:
                    with open(save_fn + "_tr.pkl", 'wb') as f:
                        pickle.dump(training_results, f, protocol=4)

            h, _ = self.actor.forward_pass(self.eval_batch[0], copy.copy(h_init), copy.copy(m_init))
            results['final_mean_h'].append(np.mean(h.numpy(), axis = (0,2)))
            results['successes'].append(k)
            
            # Reset the optimizer to facilitate training of the next model
            self.actor.reset_optimizer()

            # If specified
            if self._args.do_overwrite:
                pickle.dump(results, open(save_fn+".pkl", 'wb'), protocol=4)

        pickle.dump(results, open(save_fn + ".pkl", 'wb'))
        if self._args.save_activities:
            self.outfile.close()


def define_dependent_params(params, stim, args):

    params.n_actions = stim.n_output
    params.n_hidden = params.n_exc + params.n_inh
    params.n_bottom_up = stim.n_motion_tuned  + stim.n_fix_tuned + stim.n_cue_tuned
    params.n_motion_tuned = stim.n_motion_tuned
    params.n_cue_tuned = stim.n_cue_tuned
    params.n_fix_tuned = stim.n_fix_tuned
    params.n_top_down = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned
    params.n_top_down_hidden = args.n_top_down_hidden

    params.max_h_for_output = args.max_h_for_output
    params.top_down_trainable = args.top_down_trainable
    params.bottom_up_topology = args.bottom_up_topology
    params.top_down_overlapping = args.top_down_overlapping
    params.readout_trainable    = args.readout_trainable
    params.training_alg = args.training_alg

    return params