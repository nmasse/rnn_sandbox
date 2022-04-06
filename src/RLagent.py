import tensorflow as tf
import numpy as np
import argparse
import pickle
import os
import copy, uuid, glob, time, yaml
import matplotlib.pyplot as plt
from datetime import datetime

from . import layers
from . import analysis
from .actor import ActorRL, ActorContinuousRL
from tasks.TaskManager import TaskManager, default_tasks, TaskGym

np.set_printoptions(precision=3)

class RLAgent:
    def __init__(self, args, rnn_params):

        # Select GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args.gpu_idx], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

        self._args = args
        self._rnn_params = rnn_params
        self._args.gamma = [0.9, 0.95]

        self._args.lmbda = [0.0, 0.9]
        self._rnn_params.noise_rnn_sd = 0.0 
        self.model_type = self._args.model_type

        # This is needed for later dependencies, assuming full tasks with two RFs
        self._rnn_params.n_motion_tuned = 32 
        self._rnn_params.restrict_output_to_exc = args.restrict_output_to_exc

        tasks = default_tasks(self._args.task_set)

        if self._args.verbose:
            for k, v in vars(self._rnn_params).items():
                print(k, v)
        self._args.tasks = tasks
        self.n_tasks = len(tasks)

        # Adjust n_top_down:
        if self._args.task_set == '2stim':
            self._rnn_params.n_bottom_up = 65
        else:
            self._rnn_params.n_bottom_up = 33
        self._rnn_params.n_top_down = self.n_tasks
        self.env = TaskGym(
                    tasks,
                    args.batch_size,
                    self._rnn_params,
                    buffer_size=5000,
                    new_task_prob=1.)

        # Reset n motion tuned
        if self._args.task_set == '7tasks' or self._args.task_set == '2stim':
            self._rnn_params.n_motion_tuned = 64
        if self._args.task_set == '2stim_matching':
            self._rnn_params.n_motion_tuned = 96

        self._rnn_params = define_dependent_params(args, self._rnn_params, 
            self.env.task_manager)

        # Set up the agent actors
        self.actor      = ActorRL(args, self._rnn_params)
        self.actor_cont = ActorContinuousRL(
                                self._args,
                                self._rnn_params.n_top_down,
                                self._args.cont_action_dim,
                                self._args.action_bound)

        # Combine some variables for convenience
        self.total_n_ep = self._args.n_training_episodes + \
            self._args.n_evaluation_episodes

        if self._args.verbose:
            print('Trainable variables...')
            for v in self.actor.model.trainable_variables:
                print(v.name, v.shape)

            print(self.actor.model.summary())

    def gae_target(self, rewards, values, last_value, done):

        gamma = np.reshape(self._args.gamma, (1,2))
        lmbda = np.reshape(self._args.lmbda, (1,2))

        n_vals = values.shape[-1] # number of different time horizons to compute gae
        batch_size = values.shape[1]
        gae = np.zeros_like(values)
        gae_cumulative = np.zeros((batch_size, n_vals))
        nsteps = rewards.shape[0]

        # Compute GAE
        for k in reversed(range(nsteps)):
            if k == nsteps - 1:
                nextnonterminal = 1.0 - done[-1,:]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - done[k+1,:]
                nextvalues = values[k+1,:, :]

            delta = rewards[k, :, np.newaxis] + gamma * nextvalues * nextnonterminal[:, np.newaxis] - values[k, :, :]
            gae_cumulative = gamma * lmbda * gae_cumulative * nextnonterminal[:, np.newaxis] + delta
            gae[k,:, :] = gae_cumulative
        n_step_targets = gae + values

        return gae, n_step_targets


    def train(self, rnn_params, save_fn=None, seed=1):

        # Set up to save network
        if save_fn is None:
            save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        else:
            save_fn = os.path.join(self._args.save_path, os.path.splitext(save_fn)[0] + ".pkl")
        results = {
            'args'          : self._args,
            'rnn_params'    : rnn_params,
            'episode_scores': [[] for _ in range(self.n_tasks)],
            'episode_times' : [[] for _ in range(self.n_tasks)],
            'date'          : datetime.now(),
            'mean_h'        : [],
            'max_h'         : []
        }

        # Initialize RNN parameters
        self.actor.RNN.generate_new_weights(self._rnn_params, random_seed=seed)

        # Set up to record results
        time_steps           = np.zeros((self._args.batch_size),dtype=np.int32)
        n_completed_episodes = np.zeros((self.n_tasks))
        episode_reward       = np.zeros((self._args.batch_size), dtype=np.float32)
        current_episode_time = np.zeros((self._args.batch_size), dtype=np.int32)

        # Reset the environment
        state, cont_state, mask = self.env.reset_all()

        # Initial activity states, scores, times
        h = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), 
            dtype = np.float32)
        m = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), 
            dtype = np.float32)
        if 'stsp' in self.model_type:
            syn_x = np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype=np.float32)
            syn_u = 0.3 * np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype=np.float32)
        running_epiosde_scores = -np.ones((self.n_tasks))
        running_epiosde_times  = np.zeros((self.n_tasks))

        # Set the clock ticking
        t0 = time.time()
        for ep in range(self.total_n_ep):

            train_remain = ((self._args.n_training_episodes - ep) / \
                self._args.n_training_episodes)

            # Decrease std of continuous actor w/ training
            if ep < self._args.n_training_episodes:
                actor_cont_std = train_remain * self._args.start_action_std + \
                    (1 - train_remain) * self._args.end_action_std
                lr_multiplier = 1.
                if self._args.n_learning_rate_ramp > 0:
                    lr_multiplier = np.minimum(1, 
                        ep / self._args.n_learning_rate_ramp)
                if self._args.decay_learning_rate:
                    lr_multiplier *= train_remain
            else:
                lr_multiplier = 0.
                actor_cont_std = 1e-9
            actor_cont_std = np.float32(actor_cont_std)

            # Record all necessary information during this batch
            states, actions, values, rewards, old_policies = [], [], [], [], []
            cont_states, cont_actions, cont_old_policies = [], [], []
            activity, mod = [], []
            masks = []

            # If we're not starting tabula rasa: pick up where last ep. left off 
            if ep == 0:
                dones = [np.zeros((self._args.batch_size), dtype=np.float32)]
            else:
                dones = [dones[-1]]

            if self._args.OU_noise:
                self.actor_cont.OU.scroll_forward()

            # Follow activity for length of time horizon; record, make updates
            # to policy
            for t in range(self._args.time_horizon):

                time_steps += 1
                current_episode_time += 1
                current_task_id = self.env.task_id

                # Get continuous action for this timestep
                cont_log_policy, cont_action = \
                    self.actor_cont.get_actions(cont_state, actor_cont_std)
                cont_act = self._args.cont_action_multiplier * cont_action 

                # Use continuous action to obtain next actor choice
                next_h, next_m, log_policy, action, value = \
                    self.actor.get_actions([state, cont_act, h, m])

                # Step environment based on this action
                next_state, next_cont_state, next_mask, reward, done = \
                    self.env.step_all(action)
                episode_reward += reward

                # For episodes completed: reset environment
                is_done = np.where(done > 0)[0]
                for i in is_done:

                    if done[i]:
                        j = current_task_id[i]
                        n_completed_episodes[j] += 1
                        results['episode_scores'][j].append(episode_reward[i])
                        results['episode_times'][j].append(current_episode_time[i])
                        N = len(results['episode_scores'][j])
                        N0 = np.maximum(0, N-200)
                        running_epiosde_scores[j] = np.mean(results['episode_scores'][j][N0:])
                        running_epiosde_times[j] = np.mean(results['episode_times'][j][N0:])

                        # Reset time of episode, and reward
                        episode_reward[i] = 0.
                        current_episode_time[i] = 0
                        
                # Record all information from current env. step
                states.append(copy.copy(state))
                cont_states.append(copy.copy(cont_state))
                old_policies.append(np.float32(log_policy))
                cont_old_policies.append(np.float32(cont_log_policy))
                values.append(np.squeeze(value))
                actions.append(action)
                cont_actions.append(cont_action)
                rewards.append(reward)
                dones.append(np.float32(done))
                activity.append(copy.copy(h))
                mod.append(copy.copy(m))
                masks.append(copy.copy(mask))

                # Update next variables
                state = next_state
                cont_state = next_cont_state
                h = next_h
                m = next_m
                mask = next_mask

            # Use this information to identify parameter updates
             _, _, _, _, next_values = \
                    self.actor.get_actions([state, cont_action, h, m])

            gaes, td_targets = self.gae_target(
                np.stack(rewards, axis = 0),
                np.stack(values, axis = 0),
                np.squeeze(next_values),
                np.stack(dones, axis = 0))

            activity_r          = np.reshape(np.stack(activity, axis=0), (-1, h.shape[-1]))
            mod_r               = np.reshape(np.stack(mod, axis=0), (-1, m.shape[-1]))
            states_r            = np.reshape(np.stack(states, axis=0), (-1, state.shape[-1]))
            cont_states_r       = np.reshape(np.stack(cont_states, axis=0), (-1, cont_state.shape[-1]))
            actions_r           = np.reshape(np.stack(actions, axis=0), (-1, 1))
            cont_actions_r      = np.reshape(np.stack(cont_actions, axis=0), (-1, cont_action.shape[-1]))
            old_policies_r      = np.reshape(np.stack(old_policies, axis=0), (-1, 1))
            cont_old_policies_r = np.reshape(np.stack(cont_old_policies, axis=0), (-1, 1))
            gaes_r              = np.reshape(gaes, (-1, value.shape[-1]))
            td_targets_r        = np.reshape(td_targets, (-1, value.shape[-1]))
            masks_r             = np.reshape(masks, (-1, 1))

            if self._args.normalize_gae:
                gaes_r[:, 0:1] -= np.mean(gaes_r[:, 0:1],axis=0,keepdims=True)
                gaes_r[:, 0:1] /= (1e-8 + np.mean(gaes_r[:, 0:1],axis=0,keepdims=True))
                gaes_r[:, 0:1] = np.clip(gaes_r[:, 0:1], -1., 1.)

                #td_targets_r -= np.mean(td_targets_r,axis=0,keepdims=True)
                #td_targets_r /= (1e-8 + np.mean(td_targets_r, axis=0, keepdims=True))
                #td_targets_r = np.clip(td_targets_r, -1., 1.)

            if self._args.normalize_gae_cont:
                gaes_r[:, 1:2] -= np.mean(gaes_r[:, 1:2],axis=0,keepdims=True)
                gaes_r[:, 1:2] /= (1e-8 + np.mean(gaes_r[:, 1:2],axis=0,keepdims=True))
                gaes_r[:, 1:2] = np.clip(gaes_r[:, 1:2], -1., 1.)

            N = states_r.shape[0]
            d_norms = []
            c_norms = []
            for epoch in range(self._args.epochs):

                ind = np.random.permutation(N)
                ind = np.split(np.reshape(ind, (self._args.n_minibatches, -1)), self._args.n_minibatches, axis=0)
                for j in ind:

                    args = [copy.copy(states_r[j[0], ...]),
                        cont_actions_r[j[0], :],
                        copy.copy(activity_r[j[0], ...]),
                        copy.copy(mod_r[j[0], :]),
                        copy.copy(actions_r[j[0], :]),
                        copy.copy(gaes_r[j[0], 0:1]),
                        td_targets_r[j[0], :],
                        old_policies_r[j[0], :],
                        masks_r[j[0], :],
                        self._args.learning_rate * lr_multiplier]
                    loss, critic_loss, discrete_grad_norm = self.actor.train(*args)
                    d_norms.append(np.round(discrete_grad_norm.numpy(),4))

                    if epoch == 0:
                        loss, cont_grad_norm = self.actor_cont.train(
                            copy.copy(cont_states_r[j[0], ...]),
                            cont_actions_r[j[0], ...],
                            copy.copy(gaes_r[j[0], 1:2]),
                            cont_old_policies_r[j[0], ...],
                            masks_r[j[0], :],
                            actor_cont_std,
                            self._args.cont_learning_rate * lr_multiplier)
                        c_norms.append(np.round(cont_grad_norm.numpy(),4))

            # Print update of results
            h_exc = np.stack(activity, axis=0)[...,:self._rnn_params.n_exc]
            print(f'Epiosde {ep} | mean time {np.mean(running_epiosde_times):3.2f} '
                f'| mean h {np.mean(h_exc):3.3f}  '
                f'| max h {np.max(h_exc):3.3f}  '
                f'| time {time.time()-t0:3.2f}  '
                f'| discrete norm {discrete_grad_norm.numpy():3.2f}  '
                f'| cont norm {cont_grad_norm.numpy():3.2f}  '
                f'| critic loss {critic_loss.numpy():3.4f}')
            s = "Task scores " + " | ".join([f"{running_epiosde_scores[i]:1.3f}" for i in range(self.n_tasks)])
            s += f" | Overall: {np.mean(running_epiosde_scores):1.3f}"
            results['mean_h'].append(np.mean(h_exc))
            results['max_h'].append(np.amax(h_exc))
            print(s)
            t0 = time.time()
            if ep%20==0:
                pickle.dump(results, open(save_fn,'wb'))


def define_dependent_params(args, rnn_params, stim):

    # Make sure network shape properly specified
    rnn_params.n_actions = stim.n_output
    rnn_params.n_hidden = rnn_params.n_exc + rnn_params.n_inh
    rnn_params.n_bottom_up = stim.n_motion_tuned + stim.n_fix_tuned + stim.n_cue_tuned
    rnn_params.n_top_down = stim.n_rule_tuned
    rnn_params.n_top_down_hidden = args.cont_action_dim
    rnn_params.n_motion_tuned = stim.n_motion_tuned
    rnn_params.n_cue_tuned = stim.n_cue_tuned
    rnn_params.n_fix_tuned = stim.n_fix_tuned
    rnn_params.max_h_for_output = args.max_h_for_output
    rnn_params.cont_actor_input_dim = stim.n_rule_tuned + stim.n_fix_tuned + stim.n_cue_tuned
    
    # Some specific toggles that affect connectivity/trainability
    rnn_params.top_down_trainable   = args.top_down_trainable
    rnn_params.bottom_up_topology   = args.bottom_up_topology
    rnn_params.top_down_overlapping = args.top_down_overlapping
    rnn_params.motion_restriction   = args.motion_restriction
    rnn_params.activity_cap         = args.activity_cap
    rnn_params.readout_trainable    = args.readout_trainable 

    return rnn_params
