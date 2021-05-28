import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 5)

import torch.utils.data
import scipy.ndimage

VAE_BETA = 0.1
BATCH_SIZE = 128
EPOCHS = 400
LEARNING_RATE = 1e-3
NOISINESS = 0.0

run_path = ''

MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

INCLUDE_CLASSES = []  # empty include all


# Link to dataset if download not working
# http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip to ../data/EMNIST/raw/emnist.zip

class DatasetCustom(torch.utils.data.Dataset):
    def __init__(self, is_train):

        data_tmp = torchvision.datasets.MNIST(
            root='../data',
            train=is_train,
            # split='byclass',
            download=True
        )

        global INCLUDE_CLASSES
        if len(INCLUDE_CLASSES) == 0:
            INCLUDE_CLASSES = np.arange(len(data_tmp.classes)).tolist()

        self.data = []
        for x, y_idx in data_tmp:
            if y_idx in INCLUDE_CLASSES:
                self.data.append((x, y_idx))

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def normalize(self, data):
        data_max = data.max()
        data_min = data.min()
        if data_min != data_max:
            data = ((data - data_min) / (data_max - data_min))
        return data

    def __getitem__(self, idx):
        pil_y, label_idx = self.data[idx]
        np_y = np.array(pil_y).transpose()  # (28, 28)

        if NOISINESS > 0:
            noise = np.random.rand(*np_y.shape)
            np_x = np.where(noise < NOISINESS, 0, np_y)
        else:
            np_x = np.array(np_y)

        np_y = np.expand_dims(np_y, axis=0)  # (C, W, H)
        np_y = self.normalize(np_y)
        y = torch.FloatTensor(np_y)

        np_x = np.expand_dims(np_x, axis=0)  # (C, W, H)
        np_x = self.normalize(np_x)
        x = torch.FloatTensor(np_x)

        label = np.zeros((len(INCLUDE_CLASSES),))
        label[label_idx] = 1.0
        label = torch.FloatTensor(label)
        return x, y, label


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=4),

            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),

            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),

            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=16),

            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=16),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=32)
        )
        self.encoder_mu = torch.nn.Linear(in_features=32, out_features=32)
        self.encoder_sigma = torch.nn.Linear(in_features=32, out_features=32)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=16),

            torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=16),

            torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),

            torch.nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=7),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=8),

            torch.nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, padding=1, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(num_features=4),

            torch.nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=5),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out_flat = out.view(x.size(0), -1)

        z_sigma = torch.abs(self.encoder_sigma.forward(out_flat))  # limitation not to get negative value
        z_mu = self.encoder_mu.forward(out_flat)

        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size()).to(DEVICE)
        z = z_mu + z_sigma * eps  # reparametrization trick

        z_2d = z.view(x.size(0), -1, 1, 1)
        y_prim = self.decoder(z_2d)
        return y_prim, z, z_sigma, z_mu


def main():
    data_loader_train = torch.utils.data.DataLoader(
        dataset=DatasetCustom(is_train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset=DatasetCustom(is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )

    model = VAE()
    dummy = torch.randn((32, 1, 28, 28))
    y = model.forward(dummy)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = model.to(DEVICE)

    metrics = {}
    for stage in ['train', 'test']:
        for metric in [
            'loss, loss_rec, loss_kl'
        ]:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(1, 100):

        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}

            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'

            for x in data_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                model = model.train()

                y_prim, z, z_sigma, z_mu = model.forward(x)
                loss = 0
                # TODO loss function

                loss_rec = torch.mean((y_prim - y) ** 2)
                loss_kl = torch.mean(
                    VAE_BETA * torch.mean(-0.5 * (2 * torch.log(z_sigma + 1e-8) - z_sigma - z_mu ** 2 + 1), axis=0))
                loss = loss_rec + loss_kl

                # TODO: add to others
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                np_y_prim = y_prim.cpu().data.numpy()

                # TODO save model at best loss

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
            plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))

            plt.subplot(4, 6, j + 6)
            plt.imshow(np_y_prim[i][0].T, cmap=plt.get_cmap('Greys'))

        plt.tight_layout(pad=0.5)
        plt.show()



if __name__ == '__main__':
    main()