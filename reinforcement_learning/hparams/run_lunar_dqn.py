import os
import gym  # pip install git+https://github.com/openai/gym
import argparse
import torch
import time
import logging
import numpy as np

from csv_utils import CsvUtils

import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from reinforcement_learning.dqn_agent import DQNAgent

'''
# mac os
pip install git+https://github.com/openai/gym
pip install box2d-py
pip install pyglet

# for server
conda install -c conda-forge gym https://anaconda.org/conda-forge/GYM
conda install -c conda-forge swig https://anaconda.org/conda-forge/swig
pip install box2d-py
pip install pyglet
'''

time = int(time.time())

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cpu', type=str)
parser.add_argument('-is_render', default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=1, type=int)
parser.add_argument('-episodes', default=100, type=int)
parser.add_argument('-replay_buffer_size', default=5000, type=int)

parser.add_argument('-hidden_size', default=128, type=int)

parser.add_argument('-gamma', default=0.7, type=float)
parser.add_argument('-epsilon', default=0.99, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=500, type=int)
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


def run():
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
        env.action_space.n,
        args
    )
    is_end = False

    for e in range(args.episodes):
        s_t0 = env.reset()
        reward_total = 0
        episode_loss = []
        is_win = False
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
                '''
                if terminal reward is =100 => landed
                https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L381
                '''
                if r_t1 >= 100:
                    is_win = True
                break

        all_t.append(t)
        metrics_episode = {
            'loss': all_losses[-1],
            'score': reward_total,
            't': t,
            'e': agent.epsilon,
            'is_win': is_win
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
    env.close()


def main():
    logging.info(f'Hyperparams {args}')
    print(f'Hyperparams {args}')
    print(f'module name: {__name__}')
    print(f'parent process: {os.getppid()}')
    print(f'process id: {os.getpid()}\n')
    run()


if __name__ == '__main__':
    main()

'''
* Information about terminal states and reward formulas of Lunar Lender https://gym.openai.com/envs/LunarLander-v2/
'''
