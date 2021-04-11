import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import pad, softmax
from torchsummary import summary

import torch
from torch.optim import Adam
from torchnet.meter import AverageValueMeter

import time
from tensorboard_utills import CustomSummaryWriter
import argparse

from csv_utils import CsvUtils2

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default=f'seq-resnet', type=str)
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-is_cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-dataset_path', default=f'./datasets', type=str)
args = parser.parse_args()

MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training
DEVICE = 'cpu'
if torch.cuda.is_available() and args.is_cuda:
    DEVICE = 'cuda'
    MAX_LEN = 0

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
lr = args.learning_rate

summary_writer = CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}'
)


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


def conv_3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=out_channels)
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            conv_3x3(in_channels=self.in_channels, out_channels=self.out_channels),
            nn.ReLU(),
            conv_3x3(in_channels=self.out_channels, out_channels=self.out_channels)
        )
        self.shortcut = conv_3x3(in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.layers(x)
        return x + residual


class ResidualGate(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ResidualGate, self).__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(in_channels=in_channels, out_channels=out_channels),
            *[ResidualBlock(in_channels=out_channels, out_channels=out_channels) for _ in range(blocks - 1)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ResNet, self).__init__()
        self.input = nn.Sequential(
            # out = (28 + 2*1 - 3) / 1 + 1   (28)
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # out = (28 + 2*1 - 3) / 2 + 1   (14)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        size = [16, 32, 64, 128]
        self.gates = nn.ModuleList(
            [ResidualGate(in_channels=i[0], out_channels=i[1], blocks=2) for i in tuple(zip(size, size[1:]))]
        )
        self.fc = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        x = self.input(x)
        for gate in self.gates:
            x = gate.forward(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = softmax(x, dim=1)
        return x


def main():
    train_loader = torch.utils.data.DataLoader(
        dataset=DatasetFashionMNIST(is_train=True, dataset_path=args.dataset_path),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=DatasetFashionMNIST(is_train=False, dataset_path=args.dataset_path),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ResNet(in_channels=1, n_classes=10)
    model = model.to(DEVICE)
    summary(model, (1, 28, 28))

    optimizer = Adam(model.parameters(), lr=lr)
    meters: dict = {
        'train_loss': [],
        'test_loss': []
    }
    for epoch in range(EPOCHS):
        for loader in [train_loader, test_loader]:
            if loader == train_loader:
                meter_prefix = "train"
                model = model.train()
                torch.set_grad_enabled(True)
            else:
                meter_prefix = "test"
                model = model.eval()
                torch.set_grad_enabled(False)

            losses = AverageValueMeter()
            for x, y_idx in loader:
                # if losses.n > 10:
                #     break

                x = x.to(DEVICE)
                y_idx = y_idx.to(DEVICE)
                y_prim = model.forward(x)

                # use custom implemented cross-entropy
                # loss = -torch.mean(torch.log(y_prim + 1e-8)[torch.arange(BATCH_SIZE), y_idx])
                # print(loss)

                # convert label to one-hot encoded
                y = torch.zeros((x.size(0), 10))
                y[torch.arange(x.size(0)), y_idx] = 1.0
                y = y.to(DEVICE)

                # batch loss
                loss = -torch.mean(y * torch.log(y_prim + 1e-8))

                # loss.to('cpu').item() => single scalar value
                # loss.to('cpu').data.numpy() => matrix
                losses.add(loss.to(DEVICE).item())

                if loader == train_loader:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # losses.value is average loss of all batches
            meters[f'{meter_prefix}_loss'].append(losses.value()[0])

            # summary_writer.add_scalar(
            #     tag=f'{meter_prefix}_loss',
            #     scalar_value=losses.value()[0],
            #     global_step=epoch
            # )

            # # TODO: add Acc same as loss
            #
            # # TODO
            # d = {
            #     'last_train_loss'
            #     'best_train_loss'
            #     #     accs other meters for each train/test
            # }
            #
            # summary_writer.add_hparams(
            #     hparam_dict=args.__dict__,
            #     metric_dict=d,
            #     name=args.run_name,
            #     global_step=epoch
            # )
            #
            # # TODO: run each epoch
            # CsvUtils2.add_hparams()


if __name__ == '__main__':
    # with tf.device('/device:GPU:0'):
    main()
