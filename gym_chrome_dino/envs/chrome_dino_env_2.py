#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import base64
import io
import numpy as np
import os
from collections import deque
from PIL import Image

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_chrome_dino.game import DinoGame
from gym_chrome_dino.utils.helpers import rgba2rgb

class ChromeDinoEnv_Handcrafted(gym.Env):
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, render, accelerate, autoscale):
        self.game = DinoGame(render, accelerate)
        # observation = [obstacle xpos, obstacle width, obstacle height, rex speed]
        self.observation_space = spaces.Box(0, 1000, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.gametime_reward = 0.1
        self.gameover_penalty = -2 #-1
        self.suicide_penalty = -10
        self.redundant_jump_penalty = -10
        self.action_penalty = -0.01
        self._action_set = [0, 1, 2]

    def _observe(self):
        obstacle = self.game.obstacle()
        speed = self.game.get_speed()
        framerate = self.game.get_framerate()
        if len(obstacle) > 0:
            state = [obstacle[0]['xPos'],
                     obstacle[0]['typeConfig']['width'] * obstacle[0]['size'],
                     obstacle[0]['typeConfig']['height'],
                     speed]
        else:
            state = [self.observation_space.high[0], 0, 0, speed]
        return np.array(state, dtype=np.float32)

    def _get_current_frame(self):
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b)
        i = rgba2rgb(i)
        a = np.array(i)
        self.current_frame = a
        return self.current_frame
    
    def step(self, action):
        observation = self._observe()
        reward = 0
        if action == 1:
            self.game.press_up()
            reward += self.action_penalty
            # if observation[0] == 1000: reward += self.redundant_jump_penalty
        if action == 2:
            self.game.press_down()
            reward += self.action_penalty
        if action == 3:
            self.game.press_space()
            reward += self.action_penalty
        reward += self.gametime_reward
        done = False
        info = {'score': self.game.get_score()}
        if self.game.is_crashed():
            rex = self.game.rex()
            if not rex['jumping']: reward += self.suicide_penalty
            reward += self.gameover_penalty
            done = True
        return observation, reward, done, info
    
    def reset(self, record=False):
        self.game.restart()
        return self._observe()
    
    def render(self, mode='rgb_array', close=False):
        assert mode=='rgb_array', 'Only supports rgb_array mode.'
        return self._get_current_frame
    
    def close(self):
        self.game.close()
    
    def get_score(self):
        return self.game.get_score()
    
    def set_acceleration(self, enable):
        if enable:
            self.game.restore_parameter('config.ACCELERATION')
        else:
            self.game.set_parameter('config.ACCELERATION', 0)
    
    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

ACTION_MEANING = {
    0 : "NOOP",
    1 : "UP",
    2 : "DOWN",
    3 : "SPACE",
}