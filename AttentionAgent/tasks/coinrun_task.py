import random
import time
import gin
import gym
import gym.wrappers
from gym.wrappers import Monitor
import numpy as np

import procgen

from .gym_task import GymTask

# From https://github.com/openai/procgen/blob/615e7511b2ff8426caa54b8c5c801ffd2725b9b5/procgen/env.py#L155
JUMPS = [2, 5, 8]
NOP = 4

@gin.configurable
class CoinrunTask(GymTask):
    """Gym Coinrun-v0 task."""

    def __init__(self):
        super(CoinrunTask, self).__init__()
        self._max_steps = 0
        self._last_obs = None
        self.last_jump = 0

    def seed(self, seed):
        self._env.start_level = seed

    def _process_action(self, action):
        aw = [int(1/a) for a in action]
        #print('action weights', aw, 'sum', sum(aw))
        r = random.random()
        total = 0
        chosen = None
        for i, a in enumerate(action):
            if (total + a > r):
                chosen = i
                break
            total += a
        else:
            chosen = len(action)-1
        #print('chosen', chosen)

        # can't jump too often
        if self.jump_delay and chosen in JUMPS:
            now = time.time()
            since_jump = now - self.last_jump
            if since_jump < self.jump_delay:
                chosen = NOP
            else:
                self.last_jump = now

        # stay still from time to time
        #if random.random() < 0.2:
        #    chosen = NOP
        return chosen
    
    def _process_observation(self, observation):
        return observation
        # OJO, no estamos sacando la dif!!!!!!!!1
        if self._last_obs is None:
            self._last_obs = observation
            return observation
        delta = np.uint8(np.rint((observation - self._last_obs+255)/2 + 255))
        #print(delta, delta.shape)
        self._last_obs = observation
        return delta

    def create_task(self, **kwargs):
        if 'render' in kwargs:
            self._render = kwargs['render']
        if 'max_steps' in kwargs:
            self._max_steps = kwargs['max_steps']
        if 'logger' in kwargs:
            self._logger = kwargs['logger']
        self.jump_delay = kwargs.get('jump_delay') or None

        env_string = 'procgen:procgen-coinrun-v0'
        difficulty = kwargs['difficulty'] or 'easy'
        self._logger.info('env_string: {}'.format(env_string))
        if 'render' in kwargs:
            self._env = gym.make(env_string, distribution_mode=difficulty, render_mode="human")
            self._env.metadata["render.modes"] = ["human", "rgb_array"]
            #self._env = gym.wrappers.Monitor(env=self._env, directory="./videos", force=True)
        else:
            self._env = gym.make(env_string, distribution_mode=difficulty)
        return self

    def set_video_dir(self, video_dir):
        self._env = Monitor(
            env=self._env,
            directory=video_dir,
            video_callable=lambda x: True
        )

