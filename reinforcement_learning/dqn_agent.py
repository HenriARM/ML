import os
import glob
import numpy as np
import torch

from q_model import QModel
from replay_priority_memory import ReplayPriorityMemory


class DQNAgent:
    def __init__(self, state_size, action_size, args):
        self.device = args.device
        self.is_inference = args.is_inference

        self.batch_size = args.batch_size

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = args.gamma  # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.q_model = QModel(self.state_size, self.action_size, args.hidden_size).to(args.device)

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
                s_t0 = torch.FloatTensor(s_t0).to(self.device)
                s_t0 = s_t0.unsqueeze(dim=0)
                q_all = self.q_model.forward(s_t0)
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def replay(self):
        # decay expoloration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if not self.is_inference:
            self.optimizer.zero_grad()

        batch, replay_idxes = self.replay_memory.sample()
        s_t0, a_t0, r_t1, s_t1, is_end = zip(*batch)

        s_t0 = torch.FloatTensor(s_t0).to(self.device)
        a_t0 = torch.LongTensor(a_t0).to(self.device)
        r_t1 = torch.FloatTensor(r_t1).to(self.device)
        s_t1 = torch.FloatTensor(s_t1).to(self.device)
        is_not_end = torch.FloatTensor((np.array(is_end) == False) * 1.0).to(self.device)  # 0 or 1

        idxes = torch.arange(self.batch_size).to(self.device)  # 0, 1, 2 .. batch_size

        q_t0_all = self.q_model.forward(s_t0)
        q_t0 = q_t0_all[idxes, a_t0]

        q_t1_all = self.q_model.forward(s_t1)
        a_t1 = q_t1_all.argmax(dim=1)  # dim = 0 is batch_size
        q_t1 = q_t1_all[idxes, a_t1]

        q_t1_final = r_t1 + is_not_end * (self.gamma * q_t1)

        td_error = (q_t0 - q_t1_final) ** 2
        self.replay_memory.update_priorities(replay_idxes, td_error)

        loss = torch.mean(td_error)
        if not self.is_inference:
            loss.backward()
            self.optimizer.step()

        return loss.cpu().item()
