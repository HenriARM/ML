import gym  # pip install git+https://github.com/openai/gym
import argparse
import numpy as np
import torch
import torch.nn as nn
import random

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# !pip3 install box2d-py
# pip install pyglet

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-is_render', default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-episodes', default=10000, type=int)
parser.add_argument('-replay_buffer_size', default=5000, type=int)

# parser.add_argument('-target_update', default=3000, type=int)
parser.add_argument('-hidden_size', default=128, type=int)

parser.add_argument('-gamma', default=0.7, type=float)
parser.add_argument('-epsilon', default=0.99, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=500, type=int)

args, other_args = parser.parse_known_args()

if not torch.cuda.is_available():
    args.device = 'cpu'


class ReplayPriorityMemory:
    def __init__(self, size, batch_size, prob_alpha=1):
        self.size = size
        self.batch_size = batch_size
        self.prob_alpha = prob_alpha
        self.memory = []
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        new_priority = np.median(self.priorities) if self.memory else 1.0

        self.memory.append(transition)
        if len(self.memory) > self.size:
            del self.memory[0]
        pos = len(self.memory) - 1
        self.priorities[pos] = new_priority

    def sample(self):
        probs = np.array(self.priorities)
        if len(self.memory) < len(probs):
            probs = probs[:len(self.memory)]

        probs += 1e-8
        probs = probs ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority.item()

    def __len__(self):
        return len(self.memory)


# send state -> return q value of each next action
class Model(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Model, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.LayerNorm(normalized_shape=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=action_size)
        )

    def forward(self, s_t0):
        return self.layers.forward(s_t0)


class ReplayMemory:
    def __init__(self, size, batch_size):
        self.size = size
        self.batch_size = batch_size
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.size:
            del self.memory[0]

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.is_double = True

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = args.gamma  # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.device = args.device
        self.q_model = Model(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.q_model.parameters(),
            lr=self.learning_rate,
        )

        self.replay_memory = ReplayMemory(args.replay_buffer_size, args.batch_size)

    def act(self, s_t0):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                s_t0 = torch.FloatTensor(s_t0).to(args.device)
                s_t0 = s_t0.unsqueeze(dim=0)
                q_all = self.q_model.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self):
        # TODO: why epsilon is not getting smaller + how it decreased and went back at the start?

        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.optimizer.zero_grad()
        batch = self.replay_memory.sample()
        s_t0, a_t0, r_t1, s_t1, is_end = zip(*batch)

        s_t0 = torch.FloatTensor(s_t0).to(args.device)
        a_t0 = torch.LongTensor(a_t0).to(args.device)
        r_t1 = torch.FloatTensor(r_t1).to(args.device)
        s_t1 = torch.FloatTensor(s_t1).to(args.device)
        is_not_end = torch.FloatTensor((np.array(is_end) == False) * 1.0).to(args.device)  # 0 or 1

        idxes = torch.arange(args.batch_size).to(args.device)  # 0, 1, 2 .. batch_size

        q_t0_all = self.q_model.forward(s_t0)
        q_t0 = q_t0_all[idxes, a_t0]

        q_t1_all = self.q_model.forward(s_t1)
        a_t1 = q_t1_all.argmax(dim=1)  # dim = 0 is batch_size
        q_t1 = q_t1_all[idxes, a_t1]

        q_t1_final = r_t1 + is_not_end * (args.gamma * q_t1)
        loss = torch.mean((q_t0 - q_t1_final) ** 2)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()


# environment name
env = gym.make('LunarLander-v2')
plt.figure()

all_scores = []
all_losses = []
all_t = []

agent = DQNAgent(
    env.observation_space.shape[0],
    # first 2 are position in x axis and y axis(hieght) , other 2 are the x,y axis velocity terms,
    # lander angle and angular velocity, left and right left contact points (bool)
    env.action_space.n
)
is_end = False

for e in range(args.episodes):
    s_t0 = env.reset()
    reward_total = 0
    episode_loss = []
    for t in range(args.max_steps):
        if args.is_render and len(all_scores):  # and all_scores[-1] > 0:
            # if e % 10 == 0 and all_scores[-1] > 0:
            env.render()
        a_t0 = agent.act(s_t0)
        s_t1, r_t1, is_end, _ = env.step(a_t0)

        reward_total += r_t1

        if t == args.max_steps - 1:
            r_t1 = -100
            is_end = True

        agent.replay_memory.push(
            (s_t0, a_t0, r_t1, s_t1, is_end)
        )
        s_t0 = s_t1

        if len(agent.replay_memory) > args.batch_size:
            loss = agent.replay()
            episode_loss.append(loss)

        if is_end:
            all_scores.append(reward_total)
            all_losses.append(np.mean(episode_loss))
            break

    all_t.append(t)
    print(
        f'episode: {e}/{args.episodes} '
        f'loss: {all_losses[-1]} '
        f'score: {reward_total} '
        f't: {t} '
        f'e: {agent.epsilon}')

    plt.clf()

    plt.subplot(3, 1, 1)
    plt.ylabel('Score')
    plt.plot(all_scores)

    plt.subplot(3, 1, 2)
    plt.ylabel('Loss')
    plt.plot(all_losses)

    plt.subplot(3, 1, 3)
    plt.ylabel('Steps')
    plt.plot(all_t)

    plt.xlabel('Episode')
    plt.pause(1e-3)  # pause a bit so that plots are updated

env.close()
plt.ioff()
plt.show()

'''
* Information about terminal states and reward formulas of Lunar Lender https://gym.openai.com/envs/LunarLander-v2/
'''
