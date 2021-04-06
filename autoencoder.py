import random

import torch.nn.functional as F

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)

import torch.utils.data
import scipy.misc
import scipy.ndimage

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.data = torchvision.datasets.EMNIST(
            root='./data',
            train=is_train,
            split='byclass',
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def normalize(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_min == x_max or x_max == 0:
            return x
        return (x - x_min) / (x_max - x_min)

    def __getitem__(self, idx):
        pil_x, label_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = self.normalize(np_x)
        np_x = np.expand_dims(np_x, axis=0)

        # TODO: denoising autoencoder (add noise)
        # np_noise = np.random.random(size=np_x.shape)
        # np_x_noisy = np.where(np_noise<0.5, 0.0, np_x)
        # x = torch.FloatTensor(np_x_noisy)

        x = torch.FloatTensor(np_x)
        y = torch.FloatTensor(np_x)

        np_label = np.zeros((len(self.data.classes),))
        np_label[label_idx] = 1.0
        label = torch.FloatTensor(np_label)
        return x, y, label


ds = DatasetEMNIST(is_train=True)

data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetEMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetEMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            # TODO
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 1, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.AdaptiveAvgPool2d((28, 28))
        )

    def forward(self, x):
        z = self.encoder.forward(x)
        out = self.decoder.forward(z)
        return out



def main():
    model = AutoEncoder()
    dummy = torch.randn((32, 1, 28, 28))
    y = model.forward(dummy)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = model.to(DEVICE)

    metrics = {}
    for stage in ['train', 'test']:
        for metric in [
            'loss'
        ]:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(1, 100):

        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}

            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'

            for x, y, label in data_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                y_prim = model.forward(x)
                loss = 0  # TODO
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                np_y_prim = y_prim.cpu().data.numpy()
                x = x.cpu()
                y = y.cpu()
                np_label = label.data.numpy()
                idx_label = np.argmax(np_label, axis=1)

            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        plt.subplot(121)  # row col idx
        plts = []
        c = 0
        for key, value in metrics.items():
            value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])

        for i, j in enumerate([4, 5, 6, 16, 17, 18]):
            plt.subplot(4, 6, j)
            plt.title(f"class: {data_loader.dataset.data.classes[idx_label[i]]}")
            plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))

            plt.subplot(4, 6, j + 6)
            plt.imshow(np_y_prim[i][0].T, cmap=plt.get_cmap('Greys'))

        plt.tight_layout(pad=0.5)
        plt.show()



        # torch.save(model.state_dict(), './last-auto-encoder.pt')

    input('quit?')


if __name__ == '__main__':
    main()

# TODO: use GroupNorm
# TODO: add image normalizations?