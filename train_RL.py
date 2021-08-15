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
from actor import ActorRL, ActorContinuousRL
from TaskManager import TaskManager, TaskGym, default_tasks
import yaml
import time
from datetime import datetime
import uuid

parser = argparse.ArgumentParser('')
parser.add_argument('gpu_idx', type=int)
parser.add_argument('--n_training_episodes', type=int, default=2000)
parser.add_argument('--n_evaluation_episodes', type=int, default=50)
parser.add_argument('--n_episodes', type=int, default=2000)
parser.add_argument('--time_horizon', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--n_minibatches', type=int, default=4)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--normalize_gae', type=bool, default=False)
parser.add_argument('--normalize_gae_cont', type=bool, default=False)
parser.add_argument('--entropy_coeff', type=float, default=0.002)
parser.add_argument('--critic_coeff', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--cont_learning_rate', type=float, default=5e-5)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--decay_learning_rate', type=bool, default=True)
parser.add_argument('--clip_grad_norm', type=float, default=1.)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--training_type', type=str, default='RL')
parser.add_argument('--rnn_params_path', type=str, default="./rnn_params/stringent_params/")
parser.add_argument('--rnn_params_fn', type=str, default='full_2.yaml')
parser.add_argument('--save_path', type=str, default='./results/RL/7tasks_aug14')
parser.add_argument('--start_action_std', type=float, default=0.05)
parser.add_argument('--end_action_std', type=float, default=0.05)
parser.add_argument('--OU_noise', type=bool, default=True)
parser.add_argument('--OU_theta', type=float, default=0.15)
parser.add_argument('--OU_clip_noise', type=float, default=3.)
parser.add_argument('--max_h_for_output', type=float, default=5.)
parser.add_argument('--action_bound', type=float, default=5)
parser.add_argument('--cont_action_dim', type=int, default=64)
parser.add_argument('--disable_cont_action', type=bool, default=False)
parser.add_argument('--restrict_output_to_exc', type=bool, default=False)
parser.add_argument('--cont_action_multiplier', type=float, default=1.0)
parser.add_argument('--model_type', type=str, default='model_experimental')
parser.add_argument('--task_set', type=str, default='7tasks')
parser.add_argument('-s', '--save_fn', type=str, default=None)



args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[args.gpu_idx], 'GPU')


class Agent:
    def __init__(self, args, rnn_params):

        self._args = args
        self._rnn_params = rnn_params
        self._args.gamma = [0.9, 0.95]
        self._args.lmbda = [0., 0.9]
        print(f"Gamma: {self._args.gamma[0]}, {self._args.gamma[1]}")
        print(f"Lamnda: {self._args.lmbda[0]}, {self._args.lmbda[1]}")
        print(f"Modulation time constant {self._rnn_params.tc_modulator}")
        self._rnn_params.noise_rnn_sd = 0.0
        print(f"Setting noise to {self._rnn_params.noise_rnn_sd}")
        self._rnn_params.n_motion_tuned = 32 # This is needed for later dependencies, assuming full 7 tasks with two RFs
        
        print(f"Setting number of motion tuned to {self._rnn_params.n_motion_tuned}")
        print(f"Bottom up size {self._rnn_params.n_bottom_up}")
        self._rnn_params.restrict_output_to_exc = args.restrict_output_to_exc

        tasks = default_tasks(self._args.task_set)
        for task in tasks:
            if 'mask_duration' in task.keys():
                task['mask_duration'] = 0

        for k, v in vars(self._rnn_params).items():
            print(k, v)
        self._args.tasks = tasks
        self.n_tasks = len(tasks)

        self.env = TaskGym(
                    tasks,
                    args.batch_size,
                    self._rnn_params,
                    buffer_size=10000,
                    new_task_prob=1.)

        if self._args.task_set == '7tasks':
            self._rnn_params.n_motion_tuned = 64
            self._rnn_params.n_bottom_up = 67

        

        self.actor = ActorRL(args, self._rnn_params)

        self.actor_cont = ActorContinuousRL(
                                self._args,
                                self._rnn_params.n_top_down,
                                self._args.cont_action_dim,
                                self._args.action_bound)

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


    def train(self):

        if self._args.save_fn is None:
            save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        else:
            save_fn = os.path.join(self._args.save_path, self._args.save_fn)
        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'episode_scores': [[] for _ in range(self.n_tasks)],
            'episode_times': [[] for _ in range(self.n_tasks)],
            'date': datetime.now()
        }

        while True:

            self.actor.RNN.generate_new_weights(self._rnn_params)

            time_steps =  np.zeros((self._args.batch_size), dtype=np.int32)
            n_completed_episodes = np.zeros((self.n_tasks))
            t0 = time.time()
            episode_reward = np.zeros((self._args.batch_size), dtype=np.float32)
            current_episode_time = np.zeros((self._args.batch_size), dtype=np.int32)

            state, cont_state, mask = self.env.reset_all()

            h = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype = np.float32)
            m = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype = np.float32)
            running_epiosde_scores = -np.ones((self.n_tasks))
            running_epiosde_times = np.zeros((self.n_tasks))
            last_mean_h = 0.


            for ep in range(self._args.n_training_episodes+self._args.n_evaluation_episodes):

                if ep < self._args.n_training_episodes:
                    #alpha = np.clip(np.mean(running_epiosde_scores), 0., 1)
                    actor_cont_std = self._args.start_action_std
                    lr_multiplier = 1.
                    if self._args.n_learning_rate_ramp > 0:
                        lr_multiplier = np.minimum(1, ep / self._args.n_learning_rate_ramp)
                    if self._args.decay_learning_rate:
                        lr_multiplier *= (self._args.n_training_episodes - ep) / self._args.n_training_episodes
                else:
                    lr_multiplier = 0.
                    actor_cont_std = 1e-9
                actor_cont_std = np.float32(actor_cont_std)

                states, actions, values , rewards, old_policies = [], [], [], [], []
                cont_states, cont_actions, cont_old_policies = [], [], []

                activity, mod = [], []
                masks = []

                if ep == 0:
                    dones = [np.zeros((self._args.batch_size), dtype=np.float32)]
                else:
                    dones = [dones[-1]]

                #self.actor_cont.OU.scroll_forward()

                for t in range(self._args.time_horizon):
                    time_steps += 1
                    current_episode_time += 1
                    current_task_id = self.env.task_id

                    cont_log_policy, cont_action = self.actor_cont.get_actions(cont_state, actor_cont_std)
                    cont_act = self._args.cont_action_multiplier*cont_action
                    next_h, next_m, log_policy, action, value = self.actor.get_actions([state, cont_act, h, m])
                    next_state, next_cont_state, next_mask, reward, done = self.env.step_all(action)
                    episode_reward += reward

                    for i in range(self._args.batch_size):

                        if done[i]:
                            j = current_task_id[i]
                            n_completed_episodes[j] += 1
                            results['episode_scores'][j].append(episode_reward[i])
                            results['episode_times'][j].append(current_episode_time[i])
                            N = len(results['episode_scores'][j])
                            N0 = np.maximum(0, N-200)
                            running_epiosde_scores[j] = np.mean(results['episode_scores'][j][N0:])
                            running_epiosde_times[j] = np.mean(results['episode_times'][j][N0:])
                            episode_reward[i] = 0.
                            current_episode_time[i] = 0


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

                    state = next_state
                    cont_state = next_cont_state
                    h = next_h
                    m = next_m
                    mask = next_mask


                _, _, _, _, next_values = self.actor.get_actions([state, cont_action, h, m])

                gaes, td_targets = self.gae_target(
                    np.stack(rewards, axis = 0),
                    np.stack(values, axis = 0),
                    np.squeeze(next_values),
                    np.stack(dones, axis = 0))

                activity_r = np.reshape(np.stack(activity, axis=0), (-1, h.shape[-1]))
                mod_r = np.reshape(np.stack(mod, axis=0), (-1, m.shape[-1]))
                states_r = np.reshape(np.stack(states, axis=0), (-1, state.shape[-1]))
                cont_states_r = np.reshape(np.stack(cont_states, axis=0), (-1, cont_state.shape[-1]))
                actions_r = np.reshape(np.stack(actions, axis=0), (-1, 1))
                cont_actions_r = np.reshape(np.stack(cont_actions, axis=0), (-1, cont_action.shape[-1]))
                old_policies_r = np.reshape(np.stack(old_policies, axis=0), (-1, 1))
                cont_old_policies_r = np.reshape(np.stack(cont_old_policies, axis=0), (-1, 1))
                gaes_r = np.reshape(gaes, (-1, value.shape[-1]))
                td_targets_r = np.reshape(td_targets, (-1, value.shape[-1]))
                masks_r = np.reshape(masks, (-1, 1))


                if self._args.normalize_gae:
                    gaes_r[:, 0:1] -= np.mean(gaes_r[:, 0:1],axis=0,keepdims=True)
                    gaes_r[:, 0:1] /= (1e-8 + np.mean(gaes_r[:, 0:1],axis=0,keepdims=True))
                    gaes_r[:, 0:1] = np.clip(gaes_r[:, 0:1], -5., 5.)
                if self._args.normalize_gae_cont:
                    gaes_r[:, 1:2] -= np.mean(gaes_r[:, 1:2],axis=0,keepdims=True)
                    gaes_r[:, 1:2] /= (1e-8 + np.mean(gaes_r[:, 1:2],axis=0,keepdims=True))
                    gaes_r[:, 1:2] = np.clip(gaes_r[:, 1:2], -5., 5.)


                N = states_r.shape[0]
                d_norms = []
                c_norms = []
                for epoch in range(self._args.epochs):

                    ind = np.random.permutation(N)
                    ind = np.split(np.reshape(ind, (self._args.n_minibatches, -1)), self._args.n_minibatches, axis=0)
                    for j in ind:
                        loss, critic_loss, discrete_grad_norm = self.actor.train(
                            copy.copy(states_r[j[0], ...]),
                            cont_actions_r[j[0], :],
                            copy.copy(activity_r[j[0], ...]),
                            copy.copy(mod_r[j[0], :]),
                            copy.copy(actions_r[j[0], :]),
                            copy.copy(gaes_r[j[0], 0:1]),
                            td_targets_r[j[0], :],
                            old_policies_r[j[0], :],
                            masks_r[j[0], :],
                            self._args.learning_rate * lr_multiplier)
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


                h_exc = np.stack(activity, axis=0)[...,:self._rnn_params.n_exc]
                print(f'Epiosde {ep} | mean time {np.mean(running_epiosde_times):3.2f} '
                    f'| mean h {np.mean(h_exc):3.3f}  '
                    f'| max h {np.max(h_exc):3.3f}  '
                    f'| time {time.time()-t0:3.2f}  '
                    f'| discrete norm {discrete_grad_norm.numpy():3.2f}  '
                    f'| cont norm {cont_grad_norm.numpy():3.2f}  '
                    f'| critic loss {critic_loss.numpy():3.4f}')
                s = "Task scores " + " | ".join([f"{running_epiosde_scores[i]:1.3f}" for i in range(self.n_tasks)])
                s += f" | Overal: {np.mean(running_epiosde_scores):1.3f}"
                print(s)

                if np.mean(h_exc) > 0.15:
                    self.actor.reset_optimizer()
                    reset_run = True
                    break
                t0 = time.time()
                if ep%10==0:
                    pickle.dump(results, open(save_fn,'wb'))
            return


def define_dependent_params(args, rnn_params, stim):

    rnn_params.n_actions = stim.n_output
    rnn_params.n_hidden = rnn_params.n_exc + rnn_params.n_inh

    rnn_params.n_bottom_up = stim.n_motion_tuned + stim.n_fix_tuned + stim.n_cue_tuned
    rnn_params.n_top_down = args.cont_action_dim
    rnn_params.n_motion_tuned = stim.n_motion_tuned
    rnn_params.n_cue_tuned = stim.n_cue_tuned
    rnn_params.n_fix_tuned = stim.n_fix_tuned
    rnn_params.max_h_for_output = args.max_h_for_output
    rnn_params.cont_actor_input_dim = stim.n_rule_tuned + stim.n_fix_tuned + stim.n_cue_tuned

    print('RNN params')
    print(rnn_params)

    return rnn_params



if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

rnn_params = yaml.load(open(os.path.join(args.rnn_params_path, args.rnn_params_fn)), Loader=yaml.FullLoader)

rnn_params = argparse.Namespace(**rnn_params)
agent = Agent(args, rnn_params)
for _ in range(1):
    agent.train()
