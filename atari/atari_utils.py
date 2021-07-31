import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import random
from gym.envs.atari import AtariEnv
from baselines.common.atari_wrappers import MaxAndSkipEnv,NoopResetEnv, EpisodicLifeEnv, FireResetEnv, LazyFrames
import matplotlib.pyplot as plt
import cv2
from collections import deque



class GameWrapper:
    """Wrapper for the environment provided by Gym"""
    def __init__(self, args, noop_max=30, skip=4, episodic_life=True, frame_stack=4, scale=True, eval = False, env_names=None):

        self._args = args
        self._batch_size = args.batch_size
        self._frame_stack = frame_stack
        self._width=84
        self._height=84
        self._grayscale=True
        self._episodic_life = episodic_life
        self._eval = eval
        self._scale = scale

        if env_names is None:
            self.envs = [gym.make(args.env_name)  for _ in range(self._batch_size)]
        elif len(env_names) == self._batch_size:
            self.envs = [gym.make(name)  for name in env_names]
        elif self._batch_size % len(env_names) == 0:
            N = self._batch_size // len(env_names)
            self.envs = []
            for name in env_names:
                for _ in range(N):
                    self.envs.append(gym.make(name))
        else:
            assert False, 'Something is wrong about the env_list, or its size'

        if noop_max > 0:
            self.envs = [NoopResetEnv(env,noop_max=noop_max) for env in self.envs]
        if skip > 1:
            self.envs = [MaxAndSkipEnv(env, skip=skip) for env in self.envs]
        if episodic_life:
            self.envs = [EpisodicLifeEnv(env) for env in self.envs]
        # FireResetEnv is needed for BreakOut, but can apply to all games
        for i in range(len(self.envs)):
            if 'FIRE' in self.envs[i].unwrapped.get_action_meanings():
                self.envs[i] = FireResetEnv(self.envs[i])


    def warp_frame(self, frame):

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale and self._frame_stack == 1:
            frame = np.expand_dims(frame, -1)

        return frame


    def scale(self, frames):


        if self._scale:
            frames = np.float32(frames)
            frames /= 255.
        else:
            frames = np.uint8(frames)
        return frames

    def reset(self):
        """Resets all environments"""

        self.frames = [deque([], maxlen=self._frame_stack) for _ in range(self._batch_size)]
        raw_frames = [self.envs[i].reset() for i in range(self._batch_size)]
        self.lives = [self.envs[i].unwrapped.ale.lives()  for i in range(self._batch_size)]

        warped_frames = [self.scale(self.warp_frame(frame)) for frame in raw_frames]
        for i in range(self._batch_size):
            for _ in range(self._frame_stack):
                self.frames[i].append(warped_frames[i])

        output_frames = np.transpose(np.array(self.frames), (0,2,3,1))

        return output_frames, raw_frames

    def reset_single_env(self, agent):

        self.frames[agent] = deque([], maxlen=self._frame_stack)
        raw_frames = self.envs[agent].reset()
        self.lives[agent] = self.envs[agent].unwrapped.ale.lives()
        warped_frames = self.scale(self.warp_frame(raw_frames))
        for _ in range(self._frame_stack):
            self.frames[agent].append(warped_frames)



    def step(self, action):
        """Performs an action and observes the result"""

        raw_frames = []
        rewards = []
        terminals = []
        end_of_episode = []

        for i in range(self._batch_size):
            raw_frame, reward, done, info = self.envs[i].step(action[i])
            raw_frames.append(raw_frame)

            lives = self.envs[i].unwrapped.ale.lives()
            if self._episodic_life and lives < self.lives[i]:
                # set done to True if a life is lost
                done = True

            if done and lives>0:
                raw_frame, _, _, _ = self.envs[i].step(0)
                self.lives[i] -= 1

            frame = self.scale(self.warp_frame(raw_frame))
            self.frames[i].append(frame)

            if done and lives==0 and not self._eval:
                self.reset_single_env(i)
                end_of_episode.append(True)
            else:
                end_of_episode.append(False)

            rewards.append(reward)
            terminals.append(done)

        output_frames = np.transpose(np.array(self.frames), (0,2,3,1))

        return output_frames, np.float32(rewards), np.float32(terminals), end_of_episode, raw_frames
