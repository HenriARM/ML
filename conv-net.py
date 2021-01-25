import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchnet.meter import AverageValueMeter

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'


# (C, out_W, out_H): (1,28,28) -> (8,14,14) -> (16,7,7) -> (32,2,2) -> FC(128,1,1) -> Softmax
# (input_square - kernel + 2 * padding)/stride + 1 = output_square
# since everywhere padding=1 and stride=2
# 28 - x + 2 + 1 = 14
# 14 - x + 2 + 1 = 7
# 7 - x + 2 + 1 = 2
# 2 - x + 2 + 1 = 1


class ConvNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=8, padding=1, stride=1, kernel_size=17),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, padding=1, stride=1, kernel_size=10),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, padding=1, stride=1, kernel_size=8),
            nn.BatchNorm2d(32),
            nn.ReLU())

        # used conv1x1 before or after pooling to map in-channels to out-features, without change of W and H
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_features, padding=0, stride=1, kernel_size=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU())

        # self.fc = nn.Linear(in_features=128, out_features=out_features)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # used with Dense layers
        # out = out.reshape(out.shape[0], -1)
        # out = self.fc.forward(out)

        # used in FCN
        out = self.layer4(out)
        out = self.pool(out)
        out = out.reshape(out.shape[0], -1)

        return F.softmax(out, dim=1)


def main():
    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root='./datasets/FashionMNIST/train',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # test_set = torchvision.datasets.FashionMNIST(
    #     root='./datasets/FashionMNIST/test',
    #     train=False,
    #     download=True,
    #     transform=transforms.Compose([transforms.ToTensor()])
    # )

    EPOCHS = 10
    BATCH_SIZE = 64
    lr = 1e-4

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = None  # DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

    model = ConvNet(in_features=1, out_features=10)
    model = model.to(DEVICE)  # cpu:0, cuda:1
    # DataParallel

    optimizer = Adam(model.parameters(), lr=lr)
    meters: dict = {
        'train_loss': [],
        'test_loss': []
    }
    for epoch in range(EPOCHS):
        print("\nepoch = ", epoch)
        # for loader in [train_loader, test_loader]:
        for loader in [train_loader]:
            if loader == train_loader:
                # print("\n\ttraining:")
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

                # loss.to('cpu').item() => single scalar value
                # loss.to('cpu').data.numpy() => matrix
                losses.add(loss.to('cpu').item())

                if loader == train_loader:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            # losses.value is average loss of all batches
            meters[f'{meter_prefix}_loss'].append(losses.value()[0])
    print(meters)


if __name__ == '__main__':
    main()
