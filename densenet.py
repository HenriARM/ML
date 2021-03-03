import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data
import torch.nn.functional
from torchsummary import summary
from torch.optim import Adam

EPOCHS = 10
BATCH_SIZE = 16
lr = 1e-4
MAX_LEN = 1000
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


class DatasetFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='./datasets',
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


class Reshape(torch.nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(self.target_shape)


class TransitionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            torch.nn.AvgPool2d(
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

    def forward(self, x):
        return self.layers.forward(x)


class DenseBlock(torch.nn.Module):
    def __init__(self, in_features, num_chains):
        super().__init__()
        self.chains = torch.nn.ModuleList()

        out_features = 0
        for i in range(num_chains):
            out_features += in_features
            self.chains.add_module(f'conv{i}', torch.nn.Sequential(
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=out_features,
                    out_channels=in_features,
                    kernel_size=3,
                    padding=1,
                    stride=1)
            ))

    def forward(self, x):
        for chain in self.chains:
            x = torch.cat([x, chain.forward(x)], dim=1)
        return x


class DenseNet(torch.nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        num_channels = 16
        blocks = 4

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=7,
                stride=1,
                padding=1
            ),
            *[torch.nn.Sequential(
                DenseBlock(in_features=num_channels, num_chains=4),
                TransitionLayer(in_features=num_channels + 4 * num_channels, out_features=num_channels)
            ) for _ in range(blocks - 1)],

            # last DenseBlock is without TransitionLayer, we use Adaptive Pooling
            DenseBlock(in_features=num_channels, num_chains=4),
            # before pool do C mean (80 -> 16) with conv
            torch.nn.Conv2d(
                in_channels=num_channels + 4 * num_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Reshape(target_shape=(-1, num_channels)),
            torch.nn.Linear(in_features=num_channels, out_features=n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


def test():
    model = DenseNet(in_channels=1, n_classes=10)
    summary(model, (1, 28, 28))
    # test separate NN blocks
    # dummy = DenseBlock(in_features=8)
    # x = torch.ones((BATCH_SIZE, 8, 28, 28))
    # out = dummy.forward(x)
    # print(x)
    # print(out.shape)


def main():
    train_loader = torch.utils.data.DataLoader(
        dataset=DatasetFashionMNIST(is_train=True),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=DatasetFashionMNIST(is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = DenseNet(in_channels=1, n_classes=10)
    model = model.to(DEVICE)

    # TODO: check paper optimizer and lr
    optimizer = Adam(model.parameters(), lr=lr)
    metrics = {}
    for stage in ['train', 'test']:
        for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(EPOCHS):
        plt.clf()
        for loader in [train_loader, test_loader]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            if loader == train_loader:
                stage = "train"
                model = model.train()
                torch.set_grad_enabled(True)
            else:
                stage = "test"
                model = model.eval()
                torch.set_grad_enabled(False)

            for x, y in loader:
                x = x.to(DEVICE)  # (B,C,W,H)
                y = y.to(DEVICE)  # (B, n_classes)
                y_prim = model.forward(x)
                loss = torch.sum(-y*torch.log(y_prim + 1e-8))
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

                if loader == train_loader:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                np_y_prim = y_prim.cpu().data.numpy()
                np_y = y.cpu().data.numpy()
                idx_y = np.argmax(np_y, axis=1)
                idx_y_prim = np.argmax(np_y_prim, axis=1)
                acc = np.mean((idx_y == idx_y_prim) * 1.0)
                metrics_epoch[f'{stage}_acc'].append(acc)

            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        plt.clf()
        plts = []
        c = 0
        from scipy.ndimage import gaussian_filter1d
        for key, value in metrics.items():
            value = gaussian_filter1d(value, sigma=2)
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1
        plt.legend(plts, [it.get_label() for it in plts])
        plt.show()


if __name__ == '__main__':
    # with tf.device('/device:GPU:0'):
    # test()
    main()
