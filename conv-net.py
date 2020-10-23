import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchnet.meter import AverageValueMeter


# (C, out_W, out_H): (1,28,28) -> (8,14,14) -> (16,7,7) -> (32,2,2) -> FC(192,1,1) -> Softmax
# (input_square - kernel + 2 * padding)/stride + 1 = output_square
# since everywhere padding=1 and stride=2
# 28 - x + 2 + 1 = 14
# 14 - x + 2 + 1 = 7
# 7 - x + 2 + 1 = 2
# 2 - x + 2 + 1 = 1


class ConvNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_features, out_channels=8, padding=1, stride=1, kernel_size=17)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, padding=1, stride=1, kernel_size=10)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, padding=1, stride=1, kernel_size=8)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=128, padding=1, stride=1, kernel_size=4)
        self.fc = nn.Linear(in_features=128, out_features=out_features)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc.forward(x)
        x = F.softmax(x, dim=1)
        return x


def main():
    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root='./FashionMNIST/test',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    test_set = torchvision.datasets.FashionMNIST(
        root='./FashionMNIST/train',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    EPOCHS = 2
    BATCH_SIZE = 1000
    lr = 1e-3

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = ConvNet(in_features=1, out_features=10)
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

            epoch_loss = AverageValueMeter()
            for X, Y in loader:
                optimizer.zero_grad()
                Y_hat = model.forward(x=X)
                loss = F.cross_entropy(Y_hat, Y)
                epoch_loss.add(loss.detach().numpy())
                if loader == train_loader:
                    loss.backward()
                    optimizer.step()
            meters[f'{meter_prefix}_loss'].append(epoch_loss.value()[0])

    print(meters)


if __name__ == '__main__':
    main()
