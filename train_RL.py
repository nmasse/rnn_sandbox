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
import uuid


gpu_idx = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[gpu_idx],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4800)])

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Agent:
    def __init__(self, args, rnn_params):

        self._args = args
        self._rnn_params = rnn_params

        tasks = default_tasks()
        self.n_tasks = len(tasks)
        self.env = TaskGym(tasks, args.batch_size, buffer_size=1000, new_task_prob=1.)

        stim = TaskManager(tasks, batch_size=args.batch_size, tf2=False)
        self.dms_batch = stim.generate_batch(256, rule=0)
        self.sample_decode_time = [(300+200+300+400)//rnn_params.dt, (300+200+300+780)//rnn_params.dt]

        rnn_params, cont_actor_input_dim = define_dependent_params(args, rnn_params, self.env.task_manager)
        self.actor = ActorRL(args, rnn_params)
        self.actor_cont = ActorContinuousRL(
                                args,
                                cont_actor_input_dim,
                                self._args.cont_action_dim,
                                self._args.action_bound)

        print('Trainable variables...')
        for v in self.actor.model.trainable_variables:
            print(v.name, v.shape)

        print(self.actor.model.summary())

    def gae_target(self, rewards, values, last_value, done, gamma=0.99, lmbda=0.95):

        gamma = np.reshape([0.9, 0.99], (1,2))
        lmbda = np.reshape([0., 0.95], (1,2))


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
                #nextvalues = values[k+1,:]
                nextvalues = values[k+1,:, :]

            #delta = rewards[k, :] + gamma * nextvalues * nextnonterminal - values[k, :]
            delta = rewards[k, :, np.newaxis] + gamma * nextvalues * nextnonterminal[:, np.newaxis] - values[k, :, :]
            gae_cumulative = gamma * lmbda * gae_cumulative * nextnonterminal[:, np.newaxis] + delta
            #gae[k,:] = gae_cumulative
            gae[k,:, :] = gae_cumulative
        n_step_targets = gae + values
        """
        f,ax = plt.subplots(2,2,figsize = (10,8))
        ax[0,0].plot(rewards[:, 0],'b')
        ax[0,0].plot(done[:, 0],'r')
        ax[0,1].plot(gae[:, 0,0],'b')
        ax[0,1].plot(gae[:, 0,1],'r')
        ax[1,0].plot(values[:, 0, 0],'b')
        ax[1,0].plot(values[:, 0, 1],'r')
        ax[1,1].plot(n_step_targets[:, 0, 0],'b')
        ax[1,1].plot(n_step_targets[:, 0, 1],'r')
        plt.show()
        """

        return gae, n_step_targets


    def run_loop(self, h, m, states, cont_states, rewards, dones, current_state, current_cont_action, actor_cont_std):

        activity = []
        mod = []
        values = []

        for s, cs in zip(states, cont_states):
            _, cont_action = self.actor_cont.get_actions(cs, actor_cont_std)
            if self._args.disable_cont_action:
                cont_action *= 0.
            h, m, _, action, value = self.actor.get_actions([s, cont_action, h, m])
            values.append(value)
            activity.append(h)
            mod.append(m)

        _, _, _, _, next_values = self.actor.get_actions([current_state, current_cont_action, h, m])
        gaes, td_targets = self.gae_target(
            np.stack(rewards,axis=0),
            np.stack(values, axis = 0),
            np.squeeze(next_values),
            np.stack(dones,axis=0))

        activity_r = np.reshape(np.stack(activity, axis=0), (-1, h.shape[-1]))
        mod_r = np.reshape(np.stack(mod, axis=0), (-1, m.shape[-1]))

        return activity_r, mod_r, np.reshape(gaes, (-1, gaes.shape[-1])), np.reshape(td_targets, (-1, td_targets.shape[-1]))

    def train(self):

        save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'episode_scores': [],
            'episode_times': [],
        }


        self.actor.RNN.generate_new_weights(self._rnn_params)
        """
        h_init, m_init, h = self.actor.determine_steady_state(self.dms_batch[0])
        h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
        results['sample_decoding'] = analysis.decode_signal(
                            h.numpy(),
                            np.int32(self.dms_batch[4]),
                            self.sample_decode_time)
        sd = results['sample_decoding']
        print(f"Decoding accuracy {sd[0]:1.3f}, {sd[1]:1.3f}")
        """

        time_steps =  np.zeros((self._args.batch_size), dtype=np.int32)
        n_completed_episodes =  0
        t0 = time.time()
        episode_reward = np.zeros((self._args.batch_size), dtype=np.float32)
        current_episode_time = np.zeros((self._args.batch_size), dtype=np.int32)

        state, cont_state = self.env.reset_all()

        h = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype = np.float32)
        m = 0.05*np.ones((self._args.batch_size, self._rnn_params.n_hidden), dtype = np.float32)
        running_epiosde_scores = -1

        print(h.shape, m.shape)
        for ep in range(self._args.n_episodes):

            alpha = np.clip(running_epiosde_scores, 0., 1)

            actor_cont_std = (1-alpha) * self._args.start_action_std +  alpha * self._args.end_action_std
            actor_cont_std = np.float32(actor_cont_std)


            states, actions, values , rewards, old_policies = [], [], [], [], []
            cont_states, cont_actions, cont_values, cont_old_policies = [], [], [], []
            activity, mod = [], []

            if ep == 0:
                dones = [np.zeros((self._args.batch_size), dtype=np.float32)]
            else:
                dones = [dones[-1]]

            for t in range(self._args.time_horizon):
                time_steps += 1
                current_episode_time += 1


                cont_log_policy, cont_action = self.actor_cont.get_actions(cont_state, actor_cont_std)
                if self._args.disable_cont_action:
                    cont_action *= 0.

                if t > 0:
                    cont_action = (1-done[:,np.newaxis])*cont_action + done[:,np.newaxis]*cont_actions[-1]
                    cont_log_policy = (1-done[:,np.newaxis])*cont_log_policy + done[:,np.newaxis]*cont_old_policies[-1]

                h, m, log_policy, action, value = self.actor.get_actions([state, cont_action, h, m])
                next_state, next_cont_state, reward, done = self.env.step_all(action)
                episode_reward += reward

                for i in range(self._args.batch_size):

                    if done[i]:
                        n_completed_episodes += 1
                        results['episode_scores'].append(episode_reward[i])
                        results['episode_times'].append(current_episode_time[i])
                        N = len(results['episode_scores'])
                        N0 = np.maximum(0, N-200)
                        running_epiosde_scores = np.mean(results['episode_scores'][N0:])
                        running_epiosde_time = np.mean(results['episode_times'][N0:])
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

                state = next_state
                cont_state = next_cont_state


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

            N = states_r.shape[0]
            for epoch in range(self._args.epochs):

                ind = np.random.permutation(N)
                ind = np.split(np.reshape(ind, (self._args.n_minibatches, -1)), self._args.n_minibatches, axis=0)
                for j in ind:
                    tt=time.time()
                    loss, critic_delta = self.actor.train(
                        copy.copy(states_r[j[0], ...]),
                        cont_actions_r[j[0], :],
                        copy.copy(activity_r[j[0], ...]),
                        copy.copy(mod_r[j[0], :]),
                        copy.copy(actions_r[j[0], :]),
                        copy.copy(gaes_r[j[0], 0:1]),
                        td_targets_r[j[0], :],
                        old_policies_r[j[0], :])

            self.actor_cont.train(
                copy.copy(cont_states_r),
                cont_actions_r,
                copy.copy(gaes_r[:, 1:2]),
                cont_old_policies_r,
                actor_cont_std)
            """
            activity_r, mod_r, gaes_r, td_targets_r = self.run_loop(\
                copy.copy(activity[0]), copy.copy(mod[0]), states, \
                cont_states, rewards, dones, \
                state, cont_action, actor_cont_std)
            """


            print(f'Epiosde {ep} mean score {running_epiosde_scores:1.4f}  mean time {running_epiosde_time:3.2f} mean h {np.mean(activity):3.3f}  time {time.time()-t0:3.2f} critic delta {critic_delta.numpy():2.5f}')
            t0 = time.time()





def define_dependent_params(args, rnn_params, stim):

    rnn_params.n_input   = stim.n_input
    rnn_params.n_actions = stim.n_output
    rnn_params.n_hidden = rnn_params.n_exc + rnn_params.n_inh
    rnn_params.n_bottom_up = stim.n_motion_tuned + stim.n_cue_tuned + stim.n_fix_tuned
    rnn_params.n_motion_tuned = stim.n_motion_tuned
    rnn_params.n_top_down = args.cont_action_dim

    cont_actor_input_dim = stim.n_rule_tuned

    return rnn_params, cont_actor_input_dim




parser = argparse.ArgumentParser('')
parser.add_argument('--n_episodes', type=int, default=10000)
parser.add_argument('--time_horizon', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--n_minibatches', type=int, default=1)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--normalize_gae', type=bool, default=False)
parser.add_argument('--normalize_gae_cont', type=bool, default=True)
parser.add_argument('--entropy_coeff', type=float, default=0.001)
parser.add_argument('--critic_coeff', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--cont_learning_rate', type=float, default=5e-5)
parser.add_argument('--n_learning_rate_ramp', type=int, default=10)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--training_type', type=str, default='RL')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_mod.yaml')
parser.add_argument('--save_path', type=str, default='./results/run_270721')
parser.add_argument('--start_action_std', type=float, default=0.1)
parser.add_argument('--end_action_std', type=float, default=0.01)
parser.add_argument('--action_std_episode_anneal', type=float, default=1000)
parser.add_argument('--action_bound', type=float, default=3)
parser.add_argument('--cont_action_dim', type=int, default=16)
parser.add_argument('--disable_cont_action', type=bool, default=False)


args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()

rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)

rnn_params = argparse.Namespace(**rnn_params)
agent = Agent(args, rnn_params)
agent.train()
