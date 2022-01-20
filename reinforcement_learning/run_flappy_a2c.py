import os
import time

import argparse
import numpy as np
import torch
import torch.nn as nn
import random

import logging
import ple

from csv_utils import CsvUtils

# for server
# os.environ["SDL_VIDEODRIVER"] = "dummy"

import matplotlib
# this backend cant work with pyganme simultaneously
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dqn_agent import DQNAgent

time = int(time.time())

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cpu', type=str)
parser.add_argument('-is_render', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-episodes', default=10000, type=int)
parser.add_argument('-replay_buffer_size', default=20000, type=int)

parser.add_argument('-hidden_size', default=512, type=int)

parser.add_argument('-gamma', default=0.7, type=float)
parser.add_argument('-epsilon', default=0.99, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=100000, type=int)
parser.add_argument('-is_inference', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-is_csv', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-run_name', default=f'run_{time}', type=str)
parser.add_argument('-sequence_name', default=f'seq', type=str)

args, other_args = parser.parse_known_args()

seq_run_name = os.path.join('.', args.sequence_name, args.run_name)

# if torch.cuda.is_available():
#     args.device = 'cuda'

if not os.path.exists(args.sequence_name):
    os.makedirs(args.sequence_name)

if not os.path.exists(seq_run_name):
    os.makedirs(seq_run_name)

logging.basicConfig(level=logging.INFO, filename=os.path.join(seq_run_name, 'logs.txt'), filemode='a+',
                    format='%(asctime)-15s %(levelname)-8s %(message)s')


class ModelActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(in_features=hidden_size, out_features=action_size),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, s_t0):
        return self.layers.forward(s_t0)


class ModelCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, s_t0):
        return self.layers.forward(s_t0)


class ReplayPriorityMemory:
    def __init__(self, size, batch_size, prob_alpha=1):
        self.size = size
        self.batch_size = batch_size
        self.prob_alpha = prob_alpha
        self.memory = []
        self.Rs = []
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        new_priority = np.mean(self.priorities) if self.memory else 1.0

        self.memory.append(transition)
        self.Rs.append(transition[-1])
        if len(self.memory) > self.size:
            del self.memory[0]
            del self.Rs[0]
        pos = len(self.memory) - 1
        self.priorities[pos] = new_priority

    def sample(self):
        probs = np.array(self.priorities)
        if len(self.memory) < len(probs):
            probs = probs[:len(self.memory)]

        probs = probs - np.min(probs)

        probs += 1e-8
        probs = probs ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), args.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority.item()

    def __len__(self):
        return len(self.memory)


class A2CAgent:
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

        self.model_a = ModelActor(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.model_c = ModelCritic(self.state_size, self.action_size, args.hidden_size).to(self.device)

        self.optimizer_a = torch.optim.Adam(
            self.model_a.parameters(),
            lr=self.learning_rate,
        )
        self.optimizer_c = torch.optim.Adam(
            self.model_c.parameters(),
            lr=self.learning_rate,
        )

        self.replay_memory = ReplayPriorityMemory(args.replay_buffer_size, args.batch_size)

    def act(self, s_t0):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                s_t0 = torch.FloatTensor(s_t0).to(args.device)
                s_t0 = s_t0.unsqueeze(dim=0)
                self.model_a = self.model_a.eval()
                q_all = self.model_a.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self):
        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch, batch_idxes = self.replay_memory.sample()
        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()

        s_t0, a_t0, delta = zip(*batch)

        s_t0 = torch.FloatTensor(s_t0).to(args.device)
        a_t0 = torch.LongTensor(a_t0).to(args.device)
        delta = torch.FloatTensor(delta).to(args.device)

        v_t = self.model_c.forward(s_t0).squeeze()
        A = delta - v_t

        loss_c = A ** 2
        loss_c_mean = torch.mean(loss_c)
        loss_c_mean.backward()
        self.optimizer_c.step()

        self.model_a = self.model_a.train()
        a_all = self.model_a.forward(s_t0)
        a_t = a_all[range(len(a_t0)), a_t0]

        loss_a = -A.detach() * torch.log(a_t + 1e-8)
        loss_a_mean = torch.mean(loss_a)
        loss_a_mean.backward()
        self.optimizer_a.step()

        self.replay_memory.update_priorities(batch_idxes, loss_a)

        return loss_a_mean.item(), loss_c_mean.item()


def run():
    game = ple.games.flappybird.FlappyBird()
    # game = ple.games.snake.Snake(width=512, height=512)
    # game = ple.games.pong.Pong(width=512, height=512)
    p = ple.PLE(game, fps=30, display_screen=args.is_render)
    p.init()
    plt.figure()

    all_scores = []
    all_losses = []
    all_losses_a = []
    all_losses_c = []
    all_t = []

    agent = A2CAgent(len(p.getGameState()), len(p.getActionSet()))
    is_end = p.game_over()

    for e in range(args.episodes):
        p.reset_game()
        s_t0 = np.asarray(list(p.getGameState().values()), dtype=np.float32)
        reward_total = 0
        pipes = 0

        transitions = []
        states_t1 = []
        end_t1 = []
        for t in range(args.max_steps):
            a_t0_idx = agent.act(s_t0)
            a_t0 = p.getActionSet()[a_t0_idx]
            r_t1 = p.act(a_t0)
            is_end = p.game_over()
            s_t1 = np.asarray(list(p.getGameState().values()), dtype=np.float32)
            end_t1.append(is_end)
            reward_total += r_t1

            if r_t1 == 1.0:
                pipes += 1

            transitions.append([s_t0, a_t0_idx, r_t1])
            states_t1.append(s_t1)
            s_t0 = s_t1

            if is_end:
                all_scores.append(reward_total)
                break

        t_states_t1 = torch.FloatTensor(states_t1).to(args.device)
        v_t1 = agent.model_c.forward(t_states_t1)
        np_v_t1 = v_t1.cpu().data.numpy().squeeze()
        for t in range(len(transitions)):
            s_t0, a_t0_idx, r_t1 = transitions[t]
            is_end = end_t1[t]
            delta = r_t1
            if not is_end:
                delta = r_t1 + args.gamma * np_v_t1[t]
            agent.replay_memory.push([s_t0, a_t0_idx, delta])

        loss = loss_a = loss_c = 0
        if len(agent.replay_memory) > args.batch_size:
            loss_a, loss_c = agent.replay()
            loss = loss_a + loss_c

            all_losses.append(loss)
            all_losses_a.append(loss_a)
            all_losses_c.append(loss_c)

        all_t.append(t)

        metrics_episode = {
            'loss': loss,
            'loss_a': loss_a,
            'loss_c': loss_c,
            'score': reward_total,
            't': t,
            'e': agent.epsilon,
            'pipes': pipes
        }

        if args.is_csv is True:
            CsvUtils.add_hparams(
                sequence_dir=os.path.join('.', args.sequence_name),
                sequence_name=args.sequence_name,
                run_name=args.run_name,
                args_dict=args.__dict__,
                metrics_dict=metrics_episode,
                global_step=e
            )
        else:
            logging.info(f'episode: {e}/{args.episodes} ', metrics_episode)
            print(f'episode: {e}/{args.episodes} ', metrics_episode)

        if e % 100 == 0:
            plt.clf()

            plt.subplot(5, 1, 1)
            plt.ylabel('Score')
            plt.plot(all_scores)

            plt.subplot(5, 1, 2)
            plt.ylabel('Loss')
            plt.plot(all_losses)

            plt.subplot(5, 1, 3)
            plt.ylabel('Loss Actor')
            plt.plot(all_losses_a)

            plt.subplot(5, 1, 4)
            plt.ylabel('Loss Critic')
            plt.plot(all_losses_c)

            plt.subplot(5, 1, 5)
            plt.ylabel('Steps')
            plt.plot(all_t)

            plt.xlabel('Episode')
            plt.savefig(os.path.join(seq_run_name, f'plt-{e}.png'))
            torch.save(agent.model_c.cpu().state_dict(), os.path.join(seq_run_name, f'model-{e}-c.pt'))
            torch.save(agent.model_a.cpu().state_dict(), os.path.join(seq_run_name, f'model-{e}-a.pt'))


def main():
    logging.info(f'Hyperparams {args}')
    print(f'Hyperparams {args}')
    print(f'module name: {__name__}')
    print(f'parent process: {os.getppid()}')
    print(f'process id: {os.getpid()}\n')
    run()


if __name__ == '__main__':
    main()
