import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import pad, softmax
from torchsummary import summary

import torch
from torch.optim import Adam
from torchnet.meter import AverageValueMeter

import torchvision.models.resnet

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'


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
    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root='./datasets',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_set = torchvision.datasets.FashionMNIST(
        root='./datasets',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    EPOCHS = 10
    BATCH_SIZE = 64
    lr = 1e-4

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

    model = ResNet(in_channels=1, n_classes=10)
    summary(model, (1, 28, 28))

    optimizer = Adam(model.parameters(), lr=lr)
    meters: dict = {
        'train_loss': [],
        'test_loss': []
    }
    for epoch in range(EPOCHS):
        print("\nepoch = ", epoch)
        for loader in [train_loader, test_loader]:
            if loader == train_loader:
                print("\n\ttraining:")
                meter_prefix = "train"
                model = model.train()
                torch.set_grad_enabled(True)
            else:
                print("\n\ttesting:")
                meter_prefix = "test"
                model = model.eval()
                torch.set_grad_enabled(False)

            losses = AverageValueMeter()
            for x, y_idx in loader:
                # ?
                if losses.n > 10:
                    break

                # convert label to one-hot encoded
                y = torch.zeros((x.size(0), 10))
                y[torch.arange(x.size(0)), y_idx] = 1.0

                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_prim = model.forward(x)

                # use custom implemented cross-entropy
                # batch loss
                # loss = -torch.mean(y * torch.log(y_prim + 1e-8))
                loss = -torch.mean(torch.log(y_prim + 1e-8)[torch.arange(BATCH_SIZE), y_idx])
                # print(loss)

                # loss.to('cpu').item() => single scalar value
                # loss.to('cpu').data.numpy() => matrix
                losses.add(loss.to('cpu').item())

                if loader == train_loader:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # losses.value is average loss of all batches
            meters[f'{meter_prefix}_loss'].append(losses.value()[0])
            print(losses.value()[0])
    print(meters)


if __name__ == '__main__':
    main()
