import torch
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data
import torch.nn.functional
from scipy.ndimage import gaussian_filter1d

# TODO:? why
# from torch.utils.data import Dataset

MAX_LEN = 200
BATCH_SIZE = 16

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


class DatasetFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='./data',
            train=is_train,
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        # TODO: does it shorten data size
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: read about torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = np.expand_dims(np_x, axis=0)  # (1, W, H)
        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0
        # TODO: why we are returning this type
        return torch.FloatTensor(np_x), torch.FloatTensor(np_y)


# TODO:
data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)


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
    def __init__(self, in_features, num_chains=4):
        super().__init__()
        # TODO: use ModuleList
        self.chains = []

        out_features = 0
        for _ in range(num_chains):
            out_features += in_features
            self.chains.append(torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(out_features),
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

    def parameters(self, **kwargs):
        return reduce(lambda a, b: a + b, [list(i.parameters() for i in self.chains)])

    def to(self, device):
        for i in range(len(self.chains)):
            self.chains[i] = self.chainsp[i].to(device)


class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 16
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=7,
                stride=1,
                padding=1
            ),
            # TODO: Duplicate same thing
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels + 4 * num_channels, out_features=num_channels),

            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Reshape(target_shape=(-1, num_channels)),
            torch.nn.Linear(in_features=num_channels, out_features=10),
            torch.nn.Softmax(dim=1)
        )


BATCH_SIZE = 16
dummy = DenseBlock(in_features=8)
x = torch.ones((BATCH_SIZE, 8, 28, 28))
out = dummy.forward(x)
print(x)
print(out.shape)

model = None
# TODO: code after creating model and optimizer
metrics = {}
for stage in ['train', 'test']:
    for metric in ['loss', 'acc']:
        metrics[f'{stage}_{metric}'] = []

EPOCHS = 100
DEVICE = None

train_loader = None
test_loader = None

for epoch in range(EPOCHS):

    # TODO: ?
    plt.clf()

    print("\nepoch = ", epoch)
    for loader in [train_loader, test_loader]:

        metrics_epoch = {key: [] for key in metrics.keys()}

        if loader == train_loader:
            print("\n\ttraining:")
            stage = "train"
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            print("\n\ttesting:")
            stage = "test"
            model = model.eval()
            torch.set_grad_enabled(False)

        # losses = AverageValueMeter()

        for x, y_idx in loader:
            x = x.to(DEVICE)
            # y = y.to(DEVICE)

            y_idx = y_idx.to(DEVICE)
            y_prim = model.forward(x)

            # TODO: evalds cross entropy
            # loss = torch.sum(-y*torch.log(y_prim + 1e-8))

            # use custom implemented cross-entropy
            # loss = -torch.mean(torch.log(y_prim + 1e-8)[torch.arange(BATCH_SIZE), y_idx])
            # print(loss)

            # TODO: all convertions are done inside of custom dataset
            # convert label to one-hot encoded
            y = torch.zeros((x.size(0), 10))
            y[torch.arange(x.size(0)), y_idx] = 1.0
            y = y.to(DEVICE)

            # batch loss
            loss = -torch.mean(y * torch.log(y_prim + 1e-8))

            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            # if loader == train_loader:
            #     loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            # TODO: ?
            acc = np.mean((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                # TODO: ?
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plts = []
    c = 0
    for key, value in metrics.items():
        # TODO:?
        # value = gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.show()

    # # losses.value is average loss of all batches
    # meters[f'{meter_prefix}_loss'].append(losses.value()[0])
    # print(losses.value()[0])
