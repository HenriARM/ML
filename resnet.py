import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import pad, softmax
from torchsummary import summary

import torch
from torch.optim import Adam
from torchnet.meter import AverageValueMeter

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'


def conv_plus_norm(in_channels, out_channels, kernel_size, padding, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                  stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=out_channels)
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # ResNet conv layers try not to change image size p = (k - 1)/2, our kernel is odd
        self.layers = nn.Sequential(
            conv_plus_norm(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            conv_plus_norm(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1, stride=1)
        )
        # TODO: there is 2 ways to match spatial dimensions, conv 1x1 /2 or linear projection with data padding, no?
        # TODO: how conv 1x1 with stride 2 will match if downsamples by half?
        self.shortcut = nn.Identity()
        # if self.in_channels != self.out_channels:
        #     self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=2, stride=2)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.layers(x)
        if self.in_channels != self.out_channels:
            # all filters in resnet are even
            shift = (self.out_channels - self.in_channels) // 2
            residual = pad(residual, (0, 0, 0, 0, shift, shift, 0, 0), 'constant', 0)
        return x + residual


class ResidualGate(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ResidualGate, self).__init__()
        self.blocks = [
            ResidualBlock(in_channels=in_channels, out_channels=out_channels)
        ]
        for i in range(blocks - 1):
            self.blocks.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels))

        # TODO: what is better put layers sequential inside sequential, or use list of all layers
        #  and in forward() just loop them?
        # self.blocks = nn.Sequential(
        #     ResidualBlock(in_channels=in_channels, out_channels=out_channels),
        #     *[ResidualBlock(in_channels=(out_channels), out_channels=out_channels for _ in range(blocks - 1)]
        # )

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        # TODO: read about and use nn.ModuleList
        # TODO: still didn't get when use striding for downsampling is better than pooling?
        #  https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling/387522
        self.gate1 = nn.Sequential(
            # TODO: should be bias false, so computations get faster? bias=False
            # out = (28 + 2*2 - 3) / 1 + 1   (28)
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=0),
            # TODO: read about BatchNorm2d
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # out = (28 + 2*2 - 3) / 2 + 1   (16)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.gate2 = ResidualGate(in_channels=16, out_channels=32, blocks=2)
        self.gate3 = ResidualGate(in_channels=32, out_channels=64, blocks=2)
        self.gate4 = ResidualGate(in_channels=64, out_channels=128, blocks=2)
        self.gates = [self.gate2, self.gate3, self.gate4]

    def forward(self, x):
        x = self.gate1(x)
        for gate in self.gates:
            x = gate.forward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_features, n_classes):
        super(Decoder, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = softmax(x, dim=1)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ResNet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(in_features=128, n_classes=10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root='./datasets/FashionMNIST/test',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_set = torchvision.datasets.FashionMNIST(
        root='./datasets/FashionMNIST/train',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    EPOCHS = 1
    BATCH_SIZE = 64
    lr = 1e-4

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

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
                # convert label to one-hot encoded
                y = torch.zeros((x.size(0), 10))
                y[torch.arange(x.size(0)), y_idx] = 1.0

                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_prim = model.forward(x)

                # use custom implemented cross-entropy
                # batch loss
                loss = -torch.mean(y * torch.log(y_prim))
                print(loss)

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
