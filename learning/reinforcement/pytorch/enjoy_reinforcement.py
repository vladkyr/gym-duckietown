import ast
import argparse
import logging

import os
import numpy as np
from PIL import Image
import pyglet

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from utilsx.env import launch_env
from utilsx.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper


def _enjoy():
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    policy.load(filename="ddpg", directory="models/")

    obs = env.reset()
    done = False

    while True:
        while not done:
            action = policy.predict(np.array(obs))
            print('action', action)
            # Perform action
            obs, reward, done, _ = env.step(action)
            # img_render = env.render(mode='human')
            img_render = env.render(mode='top_down')
            img = Image.fromarray(img_render, 'RGB')
            # img.rotate(180)
            img.show()

            # env.render(mode='top_down')

        done = False
        obs = env.reset()


if __name__ == "__main__":
    _enjoy()
