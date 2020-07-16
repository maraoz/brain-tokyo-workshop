import random
import gin
import gym
import gym.wrappers
from gym.wrappers import Monitor
import numpy as np

import procgen

from .gym_task import GymTask

@gin.configurable
class CoinrunTask(GymTask):
    """Gym Coinrun-v0 task."""

    def __init__(self):
        super(CoinrunTask, self).__init__()
        self._max_steps = 0
        self._last_obs = None

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
        if 'out_of_track_cap' in kwargs:
            self._neg_reward_cap = kwargs['out_of_track_cap']
        if 'max_steps' in kwargs:
            self._max_steps = kwargs['max_steps']
        if 'logger' in kwargs:
            self._logger = kwargs['logger']

        env_string = 'procgen:procgen-coinrun-v0'
        difficulty = 'easy'
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

