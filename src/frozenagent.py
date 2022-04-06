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
from .actor import ActorSL, ActorGradientEstimator, ActorBiasEstimator
from tasks.TaskManager import TaskManager, default_tasks

tf.keras.backend.set_floatx('float32')
np.set_printoptions(precision=3)

class FrozenAgent:
    def __init__(self, args, rnn_params, param_ranges=None, sz=2496):

        # Select GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args.gpu_idx], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

        self._args = args
        self._rnn_params = rnn_params

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
        self.dec_actor = ActorSL(args, self.rnn_params, learning_type='supervised',
            n_RFs=2)
        self.grad_estimator = ActorGradientEstimator(args, self.rnn_params)
        self.bias_predictor = ActorBiasEstimator(args, self.rnn_params)

        # Define training batches
        if self._args.model_type == 'model_frozen':
            self.training_batches = [[stim.generate_batch(args.batch_size, to_exclude=[], rule=j) \
                for _ in range(args.n_stim_batches)] for j in range(self.n_tasks)]
        else:
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

        print(self.dec_actor.model.summary())

    def generate_dataset(self, h0, m0, lr, n_trains=2):

        # Set up to record data
        inst_grads, true_grads = [], []
        bias_est_inputs, new_tds = [], []

        new_bias = np.random.normal(0, 1e-2, [self._rnn_params.n_hidden,])
        self.dec_actor.update_bias(new_bias)

        # Reinitialize self.td n_trains times; 
        # train network on the first 2 tasks;
        # record all data
        for n in range(n_trains):
            print(n)
            
            for t in np.arange(self.n_tasks):

                ig_n, tg_n, bei_n, nt_n = [], [], [], []
                
                for j in range(self._args.n_training_iterations):

                    batch = self.training_batches[t][j%self._args.n_stim_batches]

                    loss, h, policy, ig, tg, old_td, new_td = \
                                self.dec_actor.train(batch, copy.copy(h0), 
                                    copy.copy(m0), lr)

                    bei = tf.concat([old_td, tg], axis=-1)

                    ig_n.append(ig)
                    tg_n.append(tg)
                    bei_n.append(bei)
                    nt_n.append(new_td)
                    print(j, loss)

                inst_grads.append(tf.stack(ig_n).numpy())
                true_grads.append(tf.stack(tg_n).numpy())
                bias_est_inputs.append(tf.stack(bei_n).numpy())
                new_tds.append(tf.stack(nt_n))

        inst_grads = np.array(inst_grads).squeeze()
        true_grads = np.array(true_grads).squeeze()
        bias_est_inputs = np.array(bias_est_inputs).squeeze()
        new_tds = np.array(new_tds).squeeze()

        return inst_grads, true_grads, bias_est_inputs, new_tds

    def train_gradient_estimator(self, ig, tg, n_epochs=500):

        for ep in range(n_epochs):
            losses = []
            for minibatch in range(ig.shape[0]):
                grad_est_loss = self.grad_estimator.train(
                    ig[minibatch], tg[minibatch], self._args.learning_rate)
                losses.append(grad_est_loss)
            print(f"Epoch {ep}: {tf.reduce_mean(tf.stack(losses))}")
        return

    def train_bias_estimator(self, bei, td, n_epochs=1000):

        for ep in range(n_epochs):
            losses = []
            for minibatch in range(bei.shape[0]):
                bias_est_loss = self.bias_predictor.train(
                        bei[minibatch], td[minibatch], 1e-5)
                losses.append(bias_est_loss)
            print(f"Epoch {ep}: {bias_est_loss}")
        return

    def train_frozen(self, rnn_params, n_networks, save_fn, original_task_accuracy=None):

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
            self.dec_actor.RNN.generate_new_weights(self._rnn_params, random_seed=k+20)

            # Determine initial activity states etc.
            h_init, m_init, h = self.dec_actor.determine_steady_state(self.training_batches[0][0][0])
            results['steady_state_h'].append(np.mean(h_init))
            h, _ = self.dec_actor.forward_pass(self.eval_batch[0], copy.copy(h_init), copy.copy(m_init))

            # 1. Build training dataset for the gradient and bias estimators:
            # train network on the first 2 tasks from 10 different initializations,
            # saving all datapoints each time; then train the 2 submodels on these
            # data, grouped together in batches; then test
            ig, tg, bei, nt = self.generate_dataset(h_init, m_init, self._args.learning_rate)
            self.train_gradient_estimator(ig, tg)
            self.train_bias_estimator(bei, nt)

            print('Starting main training loop...')
            tr_acc, tr_loss = np.zeros((self._args.n_iterations, self.n_tasks)), \
                np.zeros((self._args.n_iterations, self.n_tasks))
            
            train_accs = np.zeros((len(self.to_save), self.n_tasks))
            train_weights = defaultdict(list)
            self.dec_actor.reset_optimizer()
            self.dec_actor.opt.learning_rate.assign(1e-2)

            for t in range(self.n_tasks):
                new_bias = np.random.normal(0, 1e-2, [self._rnn_params.n_hidden,])
                self.dec_actor.update_bias(new_bias)
                for j in range(self._args.n_training_iterations):
                    t0 = time.time()

                    batch = self.training_batches[t][j%self._args.n_stim_batches]
                    learning_rate = self._args.learning_rate

                    # Run decision actor, gradient estimator, top-down updater
                    loss, h, policy, inst_grads, old_td = \
                        self.dec_actor.predict(batch, copy.copy(h_init), copy.copy(m_init))
                    grad_est = self.grad_estimator.predict(tf.expand_dims(inst_grads, 0))

                    bias_est_inputs = tf.concat([old_td, tf.squeeze(grad_est)], axis=-1)
                    new_bias = self.bias_predictor.predict(tf.expand_dims(bias_est_inputs, 0))

                    # Update current bias based on bias estimator output
                    self.dec_actor.update_bias(tf.squeeze(new_bias))
                    #grad_est = tf.clip_by_norm(grad_est,1.0)
                    #self.dec_actor.opt.apply_gradients([])
                    #self.dec_actor.opt.apply_gradients([(tf.squeeze(grad_est), self.dec_actor.td)])

                    accuracies = analysis.accuracy_SL_all_tasks(
                                            np.float32(policy),
                                            np.float32(batch[1]),
                                            np.float32(batch[2]),
                                            np.int32(batch[5]),
                                            np.arange(self.n_tasks))
                    # Need to set this up to record properly
                    tr_acc[j, t] = accuracies[t]
                    tr_loss[j, t]   = loss.numpy()
                    print(f'Iteration {j} Dec loss {tr_loss[j,t]:1.4f} Accuracy {tr_acc[j,t]:1.3f} ' + 
                        f'Mean activity {np.mean(h.numpy()):2.4f} Time {time.time()-t0:2.2f} ') 

            # Add the necessary elements to the results that we record
            results['loss'].append(tr_loss)
            results['task_accuracy'].append(tr_acc)
            

            h, _ = self.dec_actor.forward_pass(self.eval_batch[0], copy.copy(h_init), copy.copy(m_init))
            results['final_mean_h'].append(np.mean(h.numpy(), axis = (0,2)))
            results['successes'].append(k)
            
            # Reset the optimizer to facilitate training of the next model
            self.dec_actor.reset_optimizer()

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