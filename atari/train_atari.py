import sys, os
sys.path.append('/home/masse/rnn_sandbox')
import tensorflow as tf
import argparse
import numpy as np
import pickle
from atari_utils import GameWrapper
from actor import ActorContinuousRL
import matplotlib.pyplot as plt
import copy
from PIL import Image
import matplotlib.pyplot as plt
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LayerNormalization
from collections import deque
import time



gpu_idx = 2
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')

parser = argparse.ArgumentParser('')
#parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4')
parser.add_argument('--env_name', type=str, default='SpaceInvadersNoFrameskip-v4')

parser.add_argument('--max_episodes', type=int, default=50000)
parser.add_argument('--atari_env', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--learning_rate', type=float, default=2.5e-4)
parser.add_argument('--cont_learning_rate', type=float, default=2.5e-5)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--cont_clip_ratio', type=float, default=0.1)
parser.add_argument('--clip_grad_norm', type=float, default=1.)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_minibatches', type=int, default=4)
parser.add_argument('--time_horizon', type=int, default=128)
parser.add_argument('--normalize_gae', type=bool, default=True)
parser.add_argument('--normalize_gae_cont', type=bool, default=True)
parser.add_argument('--critic_coeff', type=float, default=1.)
parser.add_argument('--entropy_coeff', type=float, default=0.01)
parser.add_argument('--vae_model_path', type=str, default='saved_models/saved_vae_encoder_space_invaders')
parser.add_argument('--agent_path', type=str, default='saved_models/SpaceInvadersPPO')
parser.add_argument('--binarize_states', type=bool, default=True)
parser.add_argument('--binary_threshold', default = [35])
parser.add_argument('--hidden_dims', type=int, default=[1024, 1024, 1024, 1024, 1024, 1024])
parser.add_argument('--cont_action_dim', type=int, default=1024)
parser.add_argument('--history_save_fn', type=str, default='results/space_invaders_073121.pkl')

parser.add_argument('--start_action_std', type=float, default=0.1)
parser.add_argument('--end_action_std', type=float, default=0.01)
parser.add_argument('--OU_theta', type=float, default=0.15)
parser.add_argument('--OU_clip_noise', type=float, default=3.)
parser.add_argument('--action_std_episode_anneal', type=float, default=5000)
parser.add_argument('--cont_action_gain', type=float, default=1.)
parser.add_argument('--initialization', type=str, default='Ortho')
parser.add_argument('--disable_continuous_action', type=bool, default=False)

args = parser.parse_args('')

print('Arguments:')
for k, v in vars(args).items():
    print(k,':', v)
print()



class ActorDiscrete:
    def __init__(self, args, action_dim, action_mask):
        self._args = args
        self.action_dim = action_dim
        self.action_mask = action_mask
        self.n_envs = action_mask.shape[0]
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-05)

    def create_model(self):
        state_input = tf.keras.Input(shape=(80, 80, 4), batch_size=None,dtype=tf.float32)
        context_input = tf.keras.Input(shape=(self._args.cont_action_dim,), batch_size=None,dtype=tf.float32)
        encoder_model = tf.keras.models.load_model(self._args.vae_model_path)
        encoder_model.trainable = False
        z_mean, _, _ = encoder_model(state_input)
        init = tf.keras.initializers.Orthogonal(gain=np.sqrt(2.))
        init0 = tf.keras.initializers.Orthogonal(gain=np.sqrt(1.))

        h = z_mean
        for i in range(6):

            c = Dense(
                self._args.hidden_dims[i],
                kernel_initializer=init,
                trainable=False, \
                name=f'Context{i}')(context_input)
            h = Dense(
                self._args.hidden_dims[i],
                kernel_initializer=init,
                trainable=False, \
                name=f'Dense{i}')(h)

            h = tf.nn.elu(h + c)

        policy = Dense(
                    self.action_dim,
                    kernel_initializer=init0,
                    activation='softmax',
                    name='Policy')(h)

        critic = Dense(
                    2,
                    kernel_initializer=init0,
                    activation='linear',
                    name='Crtiic')(h)


        #h1 = tf.nn.softmax(h1, axis = -1)
        network = tf.keras.Model(inputs=[state_input, context_input], outputs=[policy, critic, h])
        return network


    def get_action(self, state):
        # Numpy
        policy, values, h = self.model.predict(state)
        actions = []
        for i in range(self._args.batch_size):
            try:
                action = np.random.choice(self.action_dim, p=policy[i,:])
            except:
                action = 0
            actions.append(action)
        actions = np.stack(actions, axis = 0)
        old_policy = [np.log(1e-8 + policy[i, actions[i]]) for i in range(self._args.batch_size)]
        old_policy = np.stack(old_policy, axis = 0)

        return actions, values, old_policy, h

    def compute_policy_loss(self, old_policy, new_policy, gaes):

        new_policy = new_policy[:, tf.newaxis]
        gaes = tf.stop_gradient(gaes)
        old_policy = tf.stop_gradient(old_policy)
        ratio = tf.math.exp(new_policy - old_policy)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)


        return surrogate

    def train(self, old_policy, states, actions, gaes, td_targets, context, learning_rate):


        actions = tf.squeeze(tf.cast(actions, tf.int32))
        actions_one_hot = tf.one_hot(actions, self.action_dim)
        learning_rate = tf.cast(learning_rate, tf.float32)

        if self._args.normalize_gae:
            gaes -= tf.reduce_mean(gaes)
            gaes /= (1e-8 + tf.math.reduce_std(gaes))
            gaes = tf.clip_by_value(gaes, -3, 3.)

        with tf.GradientTape() as tape:

            policy, values, _ = self.model([states, context], training=True)

            policy += 1e-08
            log_policy = tf.reduce_sum(actions_one_hot * tf.math.log(policy),axis=1)
            entropy = - tf.reduce_sum(policy * tf.math.log(policy), axis = 1, keepdims=True)
            policy_loss = self.compute_policy_loss(old_policy, log_policy, gaes)
            entropy_loss = self._args.entropy_coeff * entropy
            value_loss = self._args.critic_coeff * 0.5 * tf.square(tf.stop_gradient(td_targets) - values)
            loss = tf.reduce_mean(policy_loss + value_loss - entropy_loss)


        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads, 0.5)
        if not tf.math.is_nan(global_norm):
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss



class Agent:
    def __init__(self, args, env_names):


        self.env = GameWrapper(args, scale=False, env_names=env_names)
        self.batch_size = len(self.env.envs)
        self.env_names = np.unique(env_names)
        self.n_envs = len(np.unique(env_names))
        self._args = args
        self.state_dim = [84,84,4]
        self.action_dim = [env.action_space.n for env in self.env.envs]
        self.max_action_dim = np.max(self.action_dim)

        self._args.gamma = [0.99, 0.995]
        self._args.lmbda = [0.95, 0.975]
        print(f"Gamma: {self._args.gamma[0]}, {self._args.gamma[1]}")
        print(f"Lamnda: {self._args.lmbda[0]}, {self._args.lmbda[1]}")

        self.action_mask = np.zeros((self.batch_size, self.max_action_dim), dtype=np.float32)
        self.context_input = np.zeros((self.batch_size, self.n_envs), dtype=np.float32)
        self.env_index = []
        for i in range(self.batch_size):
            self.action_mask[i, self.action_dim[i]:] = 1.
            for j in range(self.n_envs):
                if env_names[i] == self.env_names[j]:
                    self.context_input[i, j] = 1.
                    self.env_index.append(j)


        self.actor_discrete = ActorDiscrete(
                                args,
                                self.max_action_dim,
                                self.action_mask)
        self.actor_continuous = ActorContinuousRL(
                                args,
                                self.n_envs,
                                args.cont_action_dim,
                                5.)


        print('Trainable variables...')
        for v in self.actor_discrete.model.trainable_variables:
            print(v.name, v.shape)

        print( self.actor_discrete.model.summary())
        print( self.actor_continuous.model.summary())


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

        score_history = []
        state, _  = self.env.reset()
        state = state[:,2:-2, 2:-2, :]

        binary_threshold = np.int32(self._args.binary_threshold)
        binary_threshold = np.reshape(binary_threshold, (self.n_envs, 1, 1, 1))

        results = {
            'epiosde_scores': [[] for _ in range(self.n_envs)],
            'time_steps': [[] for _ in range(self.n_envs)],
            'args': self._args
            }

        epiosde_raw_frames = [deque([], maxlen=4000) for _ in range(self.batch_size)]
        running_epiosde_scores = np.zeros((self.n_envs), dtype=np.float32)
        epiosde_count = 0
        time_steps =  np.zeros((self.batch_size), dtype=np.int32)
        n_completed_episodes =  np.zeros((self.n_envs), dtype=np.int32)
        make_movie = [True] * self.n_envs
        t0 = time.time()
        episode_reward = np.zeros((self.batch_size), dtype=np.float32)
        current_episode_time = np.zeros((self.batch_size), dtype=np.int32)

        state = np.float32(state - binary_threshold >=0)

        for ep in range(args.max_episodes):

            alpha = np.clip(ep / self._args.action_std_episode_anneal, 0., 1.)

            actor_continuous_std = (1-alpha) * self._args.start_action_std +  alpha * self._args.end_action_std
            actor_continuous_std = np.float32(actor_continuous_std)

            states, actions, values , rewards, old_policies, cont_actions = [], [], [], [], [], []
            cont_states, cont_actions, cont_old_policies = [], [], []

            dones = [np.zeros((self.batch_size), dtype=np.float32)]
            lr = args.learning_rate * (1. - ep/args.max_episodes)




            for t in range(args.time_horizon):

                time_steps += 1
                current_episode_time += 1
                old_cont_policy, cont_action = self.actor_continuous.get_actions(self.context_input, actor_continuous_std)
                action, value, old_policy, h = self.actor_discrete.get_action([state, cont_action])
                next_state, reward, done, end_of_episode, raw_frames = self.env.step(action)
                episode_reward += reward

                for i in range(self.batch_size):
                    epiosde_raw_frames[i].append(raw_frames[i])

                    if current_episode_time[i] >= 4000:
                        end_of_episode[i] = True
                        self.env.reset_single_env(i)

                    if end_of_episode[i]:
                        j = self.env_index[i]
                        n_completed_episodes[j] += 1
                        results['epiosde_scores'][j].append(episode_reward[i])
                        results['time_steps'][j].append(time_steps[i])
                        N = len(results['epiosde_scores'][j])
                        N0 = np.maximum(0, N-50)
                        running_epiosde_scores[j] = np.mean(results['epiosde_scores'][j][N0:])
                        episode_reward[i] = 0.
                        current_episode_time[i] = 0

                        if N % 1000 == 0:
                            make_movie[j] = True
                        if make_movie[j]:
                            save_fn = 'space_invaders_skew_' + str(N//1000) +  '.mp4'
                            animation(epiosde_raw_frames[i], save_fn)
                            make_movie[j] = False
                        epiosde_raw_frames[i].clear()

                next_state = next_state[:,2:-2, 2:-2, :]
                states.append(copy.copy(state))
                cont_states.append(copy.copy(self.context_input))
                old_policies.append(np.float32(old_policy))
                cont_old_policies.append(np.float32(old_cont_policy))
                values.append(np.squeeze(value))
                actions.append(action)
                cont_actions.append(cont_action)
                rewards.append(np.clip(reward, -1., 1.))
                dones.append(np.float32(done))

                state = next_state
                state = np.float32(state - binary_threshold >=0)

            _, next_values, _, _ = self.actor_discrete.get_action([state, cont_action])

            gaes, td_targets = self.gae_target(
                np.stack(rewards, axis = 0),
                np.stack(values, axis = 0),
                np.squeeze(next_values),
                np.stack(dones, axis = 0))


            episode_gaes = np.mean(gaes, axis=0)


            states = np.reshape(np.stack(states, axis=0), (-1, 80,80,4))
            cont_states = np.reshape(np.stack(states, axis=0), (-1, self.n_envs))
            actions = np.reshape(np.stack(actions, axis=0), (-1, 1))
            cont_actions = np.reshape(np.stack(cont_actions, axis=0), (-1, self._args.cont_action_dim))
            old_policies = np.reshape(np.stack(old_policies, axis=0), (-1, 1))
            old_cont_policy = np.reshape(cont_old_policies, (-1,1))
            gaes = np.reshape(gaes, (-1, 2))
            td_targets = np.reshape(td_targets, (-1, 2))

            N = states.shape[0]
            for epoch in range(args.epochs):

                ind = np.random.permutation(N)
                ind = np.split(np.reshape(ind, (self._args.n_minibatches, -1)), self._args.n_minibatches, axis=0)
                for j in ind:
                    self.actor_discrete.train(
                        old_policies[j[0], :],
                        copy.copy(states[j[0], ...]),
                        copy.copy(actions[j[0], :]),
                        copy.copy(gaes[j[0], 0:1]),
                        td_targets[j[0], :],
                        cont_actions[j[0], :],
                        lr)


                    for epoch in range(self._args.epochs):
                        self.actor_continuous.train(
                            old_cont_policy[j[0], :],
                            copy.copy(cont_states[j[0], ...]),
                            copy.copy(cont_actions[j[0], ...]),
                            copy.copy(gaes[j[0], 1:2]),
                            None,
                            actor_continuous_std)


            if ep%5 == 0:
                time_taken = time.time() - t0
                t0 = time.time()
                print(f'Epoch {ep} Time: {time_taken:3.2f}')
                for name, score, n in zip(self.env_names, running_epiosde_scores, n_completed_episodes):
                    print(f'{name}: {score:5.2f},   # eps:{n:5d}   action sd:{actor_continuous_std:1.3f}   mean activity:{np.mean(h):1.3f}   std activity:{np.std(h):1.3f}')
                print()
                pickle.dump(results, open(self._args.history_save_fn,'wb'))
                #self.actor_discrete.model.save(self._args.agent_path + '_actor_discrete')
                #self.actor_continuous.model.save(self._args.agent_path + '_actor_continuous')


def animation(frames, save_fn):

    for i,f in enumerate(frames):
        # Export Numpy array (of frame) to image
        im = Image.fromarray(f)
        im.save('./anim/_cont_frame{:04d}.png'.format(i))

    # Use ffmpeg to collect the images into a short animation
    os.system('rm -f ./anim/{0}'.format(save_fn))
    os.system('ffmpeg -nostats -loglevel 0 -r 25 -i ./anim/_cont_frame%04d.png ' \
    + '-vcodec libx264 -crf 25 -pix_fmt yuv420p ./anim/{0}'.format(save_fn))

    # Delete the frames generated to make way for the next batch
    os.system('rm -f ./anim/_cont*.png')


env_names = [args.env_name] * args.batch_size
agent = Agent(args, env_names)
agent.train()
