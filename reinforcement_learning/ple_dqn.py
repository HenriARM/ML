import os
import ple
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import shutil
import time
import glob
import os
import logging

from csv_utils import CsvUtils

# for server
# os.environ["SDL_VIDEODRIVER"] = "dummy"

import matplotlib
# this backend cant work with pyganme simultaneously
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

time = int(time.time())

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cpu', type=str)
parser.add_argument('-is_render', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-episodes', default=10000, type=int)
parser.add_argument('-replay_buffer_size', default=5000, type=int)

parser.add_argument('-hidden_size', default=128, type=int)

parser.add_argument('-gamma', default=0.7, type=float)
parser.add_argument('-epsilon', default=0.99, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=100000, type=int)  # specific to the game
parser.add_argument('-is_inference', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-is_csv', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-run_name', default=f'run_{time}', type=str)
parser.add_argument('-sequence_name', default=f'seq', type=str)

args, other_args = parser.parse_known_args()

seq_run_name = os.path.join(f'{args.sequence_name}', args.run_name)

if torch.cuda.is_available():
    args.device = 'cuda'

if not os.path.exists(args.sequence_name):
    os.makedirs(args.sequence_name)

if not os.path.exists(seq_run_name):
    os.makedirs(seq_run_name)

logging.basicConfig(level=logging.INFO, filename=os.path.join(seq_run_name, 'logs.txt'), filemode='a+',
                    format='%(asctime)-15s %(levelname)-8s %(message)s')



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

        indices = np.random.choice(len(self.memory), args.batch_size, p=probs)
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
        self.q_model = Model(self.state_size, self.action_size, args.hidden_size).to(args.device)

        if args.is_inference:
            ckpts = []
            for file in glob.glob(os.path.join(seq_run_name, '*.pt')):
                ckpts.append(file.split('-')[-1].split('.')[0])
            model_state = torch.load(os.path.join(seq_run_name, f'model-{max(ckpts)}.pt'))
            self.q_model.load_state_dict(model_state)
            self.q_model = self.q_model.eval().to(args.device)
            logging.info(f'Model model-{max(ckpts)}.pt is loaded')

        self.optimizer = torch.optim.Adam(
            self.q_model.parameters(),
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
                q_all = self.q_model.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self):
        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if not args.is_inference:
            self.optimizer.zero_grad()

        batch, replay_idxes = self.replay_memory.sample()
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

        td_error = (q_t0 - q_t1_final) ** 2
        self.replay_memory.update_priorities(replay_idxes, td_error)

        loss = torch.mean(td_error)
        if not args.is_inference:
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

    agent = DQNAgent(len(p.getGameState()), len(p.getActionSet()))
    is_end = p.game_over()

    # best_terminal_time = 0
    for e in range(args.episodes):
        p.reset_game()
        s_t0 = np.asarray(list(p.getGameState().values()), dtype=np.float32)
        reward_total = 0
        episode_loss = []
        for t in range(args.max_steps):
            a_t0_idx = agent.act(s_t0)
            a_t0 = p.getActionSet()[a_t0_idx]
            r_t1 = p.act(a_t0)
            is_end = p.game_over()
            s_t1 = np.asarray(list(p.getGameState().values()), dtype=np.float32)

            reward_total += r_t1
            if r_t1 != 0:
                pass

            if t == args.max_steps - 1:
                r_t1 = -100
                is_end = True

            agent.replay_memory.push(
                (s_t0, a_t0_idx, r_t1, s_t1, is_end)
            )
            s_t0 = s_t1

            if len(agent.replay_memory) > args.batch_size:
                loss = agent.replay()
                episode_loss.append(loss)

            if is_end:
                all_scores.append(reward_total)
                all_losses.append(np.mean(episode_loss))
                # if t > best_terminal_time:
                #     best_terminal_time = t
                break

        all_t.append(t)

        metrics_episode = {
            'loss': all_losses[-1],
            'score': reward_total,
            't': t,
            'e': agent.epsilon
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
            torch.save(agent.q_model.cpu().state_dict(), os.path.join(seq_run_name, f'model-{e}.pt'))


def main():
    logging.info(f'Hyperparams {args}')
    print(f'Hyperparams {args}')
    print(f'module name: {__name__}')
    print(f'parent process: {os.getppid()}')
    print(f'process id: {os.getpid()}\n')
    run()


if __name__ == '__main__':
    main()
