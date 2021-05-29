import numpy as np
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import pad, softmax
from torchsummary import summary
import os

import torch
from torchnet.meter import AverageValueMeter

import time
from tensorboard_utills import CustomSummaryWriter
import argparse

from csv_utils import CsvUtils
from resnet import ResNet

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--run_name', default=f'run_{int(time.time())}', type=str)
parser.add_argument('--sequence_name', default=f'seq', type=str)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--is_cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--is_csv', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--dataset_path', default=f'./datasets', type=str)
args = parser.parse_args()

MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training
DEVICE = 'cpu'
if torch.cuda.is_available() and args.is_cuda:
    DEVICE = 'cuda'
    MAX_LEN = 0

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

summary_writer = CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}'
)


def acc(y_prim, y):
    np_y_prim = y_prim.cpu().data.numpy()
    np_y = y.cpu().data.numpy()
    idx_y = np.argmax(np_y, axis=1)
    idx_y_prim = np.argmax(np_y_prim, axis=1)
    return np.mean((idx_y == idx_y_prim) * 1.0)


class DatasetFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train, dataset_path):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root=dataset_path,
            train=is_train,
            download=True
        )

    def __len__(self):
        # len is called before iterating data loader
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        # PIL image is returned
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = np.expand_dims(np_x, axis=0)  # (1, W, H)
        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0
        # type(torch.tensor(np_x)) - <class 'torch.Tensor'>, torch.tensor(np_x).dtype - torch.uint8
        # type(torch.FloatTensor(np_x)) - <class 'torch.Tensor'>, torch.FloatTensor(np_x).dtype - torch.float32
        return torch.FloatTensor(np_x), torch.FloatTensor(np_y)


def main():
    data_loader_train = torch.utils.data.DataLoader(
        dataset=DatasetFashionMNIST(is_train=True, dataset_path=args.dataset_path),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=DatasetFashionMNIST(is_train=False, dataset_path=args.dataset_path),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ResNet(in_channels=1, n_classes=10)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metrics = {}
    for stage in ['train', 'test']:
        for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(EPOCHS):
        metrics_epoch = {key: [] for key in metrics.keys()}
        for data_loader in [data_loader_train, data_loader_test]:
            stage = 'train'
            torch.set_grad_enabled(True)
            if data_loader == data_loader_test:
                stage = 'test'
                torch.set_grad_enabled(False)

            # inference
            for x, y in data_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_prim = model.forward(x)
                loss = -torch.mean(y * torch.log(y_prim + 1e-8))
                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # calculate metrics per batch
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f
                metrics_epoch[f'{stage}_acc'].append(acc(y_prim, y))

        # calculate metrics per epoch
        metrics_epoch_str = []
        for key in metrics_epoch.keys():
            metrics_epoch[key] = np.mean(metrics_epoch[key])
            metrics_epoch_str.append(f'{key}: {round(metrics_epoch[key], 2)}')
        summary_writer.flush()
        print(f'epoch: {epoch} {" ".join(metrics_epoch_str)}')

        # add hparams
        summary_writer.add_hparams(
            hparam_dict=args.__dict__,
            metric_dict=metrics_epoch,
            name=args.run_name,
            global_step=epoch
        )
        if args.is_csv is True:
            CsvUtils.add_hparams(
                sequence_dir=os.path.join('.', f'{args.sequence_name}-csv'),
                sequence_name=args.sequence_name,
                run_name=args.run_name,
                args_dict=args.__dict__,
                metrics_dict=metrics_epoch,
                global_step=epoch
            )
        # append metrics per epoch to global metrics
        for key in metrics_epoch.keys():
            metrics[key].append(metrics_epoch[key])
        summary_writer.flush()
    summary_writer.close()


if __name__ == '__main__':
    main()
