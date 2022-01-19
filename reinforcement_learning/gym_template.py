import gym  # pip install git+https://github.com/openai/gym
import argparse
import numpy as np
import torch
import torch.nn as nn
import random

parser = argparse.ArgumentParser()
parser.add_argument('-is_render', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-episodes', default=10000, type=int)
parser.add_argument('-max_steps', default=500, type=int)
args, other_args = parser.parse_known_args()


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, s_t0):
        return env.action_space.sample()


# environment name
env = gym.make('LunarLander-v2')

agent = DQNAgent(
    env.observation_space.shape[0],
    env.action_space.n
)
is_end = False

for e in range(args.episodes):
    s_t0 = env.reset()
    for t in range(args.max_steps):
        if args.is_render and len(all_scores):
            env.render()
        a_t0 = agent.act(s_t0)
        s_t1, r_t1, is_end, _ = env.step(a_t0)
        s_t0 = s_t1
        if is_end:
            break
    print(f'episode: {e}/{args.episodes} ')
env.close()
