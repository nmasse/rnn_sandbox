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


gpu_idx = 2
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
"""
tf.config.experimental.set_virtual_device_configuration(
    gpus[gpu_idx],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4800)])
"""
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Agent:
    def __init__(self, args, rnn_params):

        self._args = args
        self._rnn_params = rnn_params
        self._args.gamma = [0.95, 0.99]
        self._args.lmbda = [0., 0.95]
        print(f"Gamma :{self._args.gamma[0]}, {self._args.gamma[1]}")
        print(f"Lamnda :{self._args.lmbda[0]}, {self._args.lmbda[1]}")


        tasks = default_tasks()
        self.n_tasks = len(tasks)
        print(f'Number of tasks {self.n_tasks}')
        stim = TaskManager(tasks, batch_size=args.batch_size, tf2=False)
        rnn_params, cont_actor_input_dim = define_dependent_params(args, rnn_params, stim)

        self.env = TaskGym(
                    tasks,
                    args.batch_size,
                    rnn_params.n_bottom_up,
                    cont_actor_input_dim,
                    buffer_size=5000,
                    new_task_prob=1.)


        self.dms_batch = stim.generate_batch(256, rule=0, include_test=True)
        #self.sample_decode_time = [(300+200+300+200)//rnn_params.dt, (300+200+300+280)//rnn_params.dt]
        #self.sample_decode_time = np.arange(0,300+200+300+600+300,100) // 20
        #self.sample_decode_time = np.arange(800,1000,200)

        self.actor = ActorRL(args, rnn_params)

        print('cont_actor_input_dim', cont_actor_input_dim)
        self.actor_cont = ActorContinuousRL(
                                args,
                                cont_actor_input_dim,
                                self._args.cont_action_dim,
                                self._args.action_bound)

        print('Trainable variables...')
        for v in self.actor.model.trainable_variables:
            print(v.name, v.shape)

        print(self.actor.model.summary())

    def gae_target(self, rewards, values, last_value, done, plot_fig=False, gamma=[0.9, 0.99], lmbda=[0., 0.9]):

        gamma = np.reshape(gamma, (1,2))
        lmbda = np.reshape(lmbda, (1,2))

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

        if plot_fig:
            f,ax = plt.subplots(2,2,figsize = (10,8))
            ax[0,0].plot(rewards[:, 0],'b')
            ax[0,0].plot(done[:, 0],'r')
            #ax[0,0].plot(actions[:, 0],'g')
            ax[0,1].plot(gae[:, 0,0],'b')
            ax[0,1].plot(gae[:, 0,1],'r')
            ax[1,0].plot(values[:, 0, 0],'b')
            ax[1,0].plot(values[:, 0, 1],'r')
            ax[1,1].plot(n_step_targets[:, 0, 0],'b')
            ax[1,1].plot(n_step_targets[:, 0, 1],'r')
            plt.show()


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
            np.stack(dones,axis=0),
            gamma=self._args.gamma,
            lmbda=self._args.lmbda)

        activity_r = np.reshape(np.stack(activity, axis=0), (-1, h.shape[-1]))
        mod_r = np.reshape(np.stack(mod, axis=0), (-1, m.shape[-1]))

        return activity_r, mod_r, np.reshape(gaes, (-1, gaes.shape[-1])), np.reshape(td_targets, (-1, td_targets.shape[-1]))

    def train(self):

        save_fn = os.path.join(self._args.save_path, 'results_'+str(uuid.uuid4())+'.pkl')
        results = {
            'args': self._args,
            'rnn_params': rnn_params,
            'episode_scores': [[] for _ in range(self.n_tasks)],
            'episode_times': [[] for _ in range(self.n_tasks)],
        }


        self.actor.RNN.generate_new_weights(self._rnn_params)

        h_init, m_init, h = self.actor.determine_steady_state(self.dms_batch[0])
        h, _ = self.actor.forward_pass(self.dms_batch[0], copy.copy(h_init), copy.copy(m_init))
        """
        plt.imshow(self.dms_batch[0][0,:,:], aspect='auto')
        plt.colorbar()
        plt.show()
        plt.imshow(h[0,:,:], aspect='auto')
        plt.colorbar()
        plt.show()
        plt.plot(np.mean(h,axis=(0,2)))
        plt.show()
        print('here')
        print(np.int32(self.dms_batch[4]))

        results['sample_decoding'] = analysis.decode_signal(
                            h.numpy(),
                            np.int32(self.dms_batch[4]),
                            self.sample_decode_time)
        results['test_decoding'] = analysis.decode_signal(
                            h.numpy(),
                            np.int32(self.dms_batch[6]),
                            self.sample_decode_time)
        m = np.int32([i==j for i,j in zip(self.dms_batch[4], self.dms_batch[6])])
        results['match_decoding'] = analysis.decode_signal(
                            h.numpy(),
                            m,
                            self.sample_decode_time)
        sd = results['sample_decoding']
        td = results['test_decoding']
        md = results['match_decoding']
        plt.plot(sd,'b')
        plt.plot(td,'r')
        plt.plot(md,'g')
        plt.show()
        print(f"Decoding accuracy {sd[0]:1.3f}, {sd[1]:1.3f}")
        """

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



        for ep in range(self._args.n_episodes):

            alpha = np.clip(np.mean(running_epiosde_scores), 0., 1)

            actor_cont_std = (1-alpha) * self._args.start_action_std +  alpha * self._args.end_action_std
            actor_cont_std = np.float32(actor_cont_std)


            states, actions, values , rewards, old_policies = [], [], [], [], []
            cont_states, cont_actions, cont_values, cont_old_policies = [], [], [], []

            activity, mod = [], []
            masks = []

            if ep == 0:
                dones = [np.zeros((self._args.batch_size), dtype=np.float32)]
            else:
                dones = [dones[-1]]

            for t in range(self._args.time_horizon):
                time_steps += 1
                current_episode_time += 1
                current_task_id = self.env.task_id

                cont_log_policy, cont_action = self.actor_cont.get_actions(cont_state, actor_cont_std)

                next_h, next_m, log_policy, action, value = self.actor.get_actions([state, cont_action, h, m])
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
                        old_policies_r[j[0], :],
                        masks_r[j[0], :])

                    self.actor_cont.train(
                        copy.copy(cont_states_r[j[0], ...]),
                        cont_actions_r[j[0], ...],
                        copy.copy(gaes_r[j[0], 1:2]),
                        cont_old_policies_r[j[0], ...],
                        masks_r[j[0], :],
                        actor_cont_std)

                """
                if epoch < self._args.epochs - 1:
                    activity_r, mod_r, gaes_r, td_targets_r = self.run_loop(\
                        copy.copy(activity[0]), copy.copy(mod[0]), states, \
                        cont_states, rewards, dones, \
                        state, cont_action, actor_cont_std)
                """


            print(f'Epiosde {ep} mean time {np.mean(running_epiosde_times):3.2f} mean h {np.mean(activity):3.3f}  time {time.time()-t0:3.2f}')
            s = "Task scores " + " | ".join([f"{running_epiosde_scores[i]:1.3f}" for i in range(self.n_tasks)])
            s += f" | Overal: {np.mean(running_epiosde_scores):1.3f}"
            print(s)
            t0 = time.time()





def define_dependent_params(args, rnn_params, stim):

    rnn_params.n_input   = stim.n_input
    rnn_params.n_actions = stim.n_output
    rnn_params.n_hidden = rnn_params.n_exc + rnn_params.n_inh
    #rnn_params.n_bottom_up = stim.n_motion_tuned + stim.n_fix_tuned
    rnn_params.n_bottom_up = stim.n_motion_tuned + stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned
    rnn_params.n_motion_tuned = stim.n_motion_tuned
    rnn_params.n_top_down = args.cont_action_dim

    #cont_actor_input_dim = stim.n_rule_tuned + stim.n_cue_tuned
    cont_actor_input_dim = stim.n_rule_tuned + stim.n_cue_tuned + stim.n_fix_tuned

    return rnn_params, cont_actor_input_dim




parser = argparse.ArgumentParser('')
parser.add_argument('--n_episodes', type=int, default=1000)
parser.add_argument('--time_horizon', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--n_minibatches', type=int, default=4)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--normalize_gae', type=bool, default=False)
parser.add_argument('--normalize_gae_cont', type=bool, default=True)
parser.add_argument('--entropy_coeff', type=float, default=0.002)
parser.add_argument('--critic_coeff', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--cont_learning_rate', type=float, default=1e-4)
parser.add_argument('--n_learning_rate_ramp', type=int, default=10)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--training_type', type=str, default='RL')
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/good_params.yaml')
parser.add_argument('--save_path', type=str, default='./results/test')
parser.add_argument('--start_action_std', type=float, default=0.1)
parser.add_argument('--end_action_std', type=float, default=0.001)
parser.add_argument('--action_std_episode_anneal', type=float, default=1000)
parser.add_argument('--action_bound', type=float, default=3)
parser.add_argument('--cont_action_dim', type=int, default=128)
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
