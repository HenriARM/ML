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
parser.add_argument('-is_render', default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-episodes', default=10000, type=int)
parser.add_argument('-replay_buffer_size', default=10000, type=int)

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


class Model(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Model, self).__init__()

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

        param_count = np.sum([np.prod(np.array(it.size())) for it in self.parameters()])
        print(f'param_count: {param_count}')  # at least 100k params

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


class PGAgent:
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
        self.p_model = Model(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.p_model.parameters(),
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
                self.p_model = self.p_model.eval()
                q_all = self.p_model.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self):
        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch, batch_idxes = self.replay_memory.sample()
        self.optimizer.zero_grad()

        s_t0, a_t0, R = zip(*batch)

        s_t0 = torch.FloatTensor(s_t0).to(args.device)
        a_t0 = torch.LongTensor(a_t0).to(args.device)
        R = torch.FloatTensor(R).to(args.device)

        self.p_model = self.p_model.train()
        a_all_probs = self.p_model.forward(s_t0)

        a_t = a_all_probs[range(len(a_t0)), a_t0]
        loss = -R * torch.log(a_t + 1e-8)

        self.replay_memory.update_priorities(batch_idxes, loss)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()


def run():
    game = ple.games.flappybird.FlappyBird()
    # game = ple.games.snake.Snake(width=512, height=512)
    # game = ple.games.pong.Pong(width=512, height=512)
    p = ple.PLE(game, fps=30, display_screen=args.is_render)
    p.init()
    plt.figure()

    all_scores = []
    all_losses = []
    all_t = []

    agent = PGAgent(len(p.getGameState()), len(p.getActionSet()))
    is_end = p.game_over()

    for e in range(args.episodes):
        p.reset_game()
        s_t0 = np.asarray(list(p.getGameState().values()), dtype=np.float32)
        reward_total = 0
        pipes = 0

        transitions = []
        for t in range(args.max_steps):
            a_t0_idx = agent.act(s_t0)
            a_t0 = p.getActionSet()[a_t0_idx]
            r_t1 = p.act(a_t0)
            is_end = p.game_over()
            s_t1 = np.asarray(list(p.getGameState().values()), dtype=np.float32)
            reward_total += r_t1

            if r_t1 == 1.0:
                pipes += 1

            if t == args.max_steps - 1:
                r_t1 = -100
                is_end = True

            transitions.append([s_t0, a_t0_idx, r_t1])
            s_t0 = s_t1

            if is_end:
                all_scores.append(reward_total)
                break

        for t in range(len(transitions)):
            R = 0
            for t_c, (s_t0, a_t0_idx, r_t) in enumerate(transitions[t:]):
                R += args.gamma ** t_c * r_t

            s_t0, a_t0_idx, r_t1 = transitions[t]
            tr = [s_t0, a_t0_idx, R]
            agent.replay_memory.push(tr)

        loss = 0
        if len(agent.replay_memory) > args.batch_size:
            loss = agent.replay()
            all_losses.append(loss)

        all_t.append(t)

        metrics_episode = {
            'loss': loss,
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


        if e % 100 == 0 and not args.is_inference:
            # save logs, graphics and weights during training
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
            plt.savefig(os.path.join(seq_run_name, f'plt-{e}.png'))
            torch.save(agent.p_model.cpu().state_dict(), os.path.join(seq_run_name, f'model-{e}.pt'))




def main():
    logging.info(f'Hyperparams {args}')
    print(f'Hyperparams {args}')
    print(f'module name: {__name__}')
    print(f'parent process: {os.getppid()}')
    print(f'process id: {os.getpid()}\n')
    run()
    # run(agent_class=DQNAgent, seq_run_name=seq_run_name, args=args)


if __name__ == '__main__':
    main()
