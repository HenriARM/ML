"""
Modern Autoencoder
"""
import scipy
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary

plt.rcParams["figure.figsize"] = (12, 5)

import torch.utils.data
import scipy.misc
import scipy.ndimage

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_LEN = 200  # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


def noise(img):
    ch, row, col = img.shape
    # mean, variance ** sigma
    gauss = np.random.normal(0, 0.1 ** 0.5, (ch, row, col))
    return img + gauss


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max or x_max == 0:
        return x
    return (x - x_min) / (x_max - x_min)


class DatasetEMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.data = torchvision.datasets.EMNIST(
            root='./datasets',
            train=is_train,
            split='byclass',
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        pil_x, label_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = normalize(np_x)
        np_x = np.expand_dims(np_x, axis=0)

        # Denoising  autoencoder
        np_x_noise = noise(np_x)
        x = torch.FloatTensor(np_x_noise)
        y = torch.FloatTensor(np_x)

        np_label = np.zeros((len(self.data.classes)))
        np_label[label_idx] = 1.0
        label = torch.FloatTensor(np_label)
        return x, y, label


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 4, 8, 16, 32]
        self.encoder = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.GroupNorm(num_groups=2, num_channels=ch_out),

                torch.nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.GroupNorm(num_groups=2, num_channels=ch_out),
                torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0)
            ) for ch_in, ch_out in zip(channels[:-1], channels[1:-1])],

            # last conv block is without pooling
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=3, stride=1,
                                padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.GroupNorm(num_groups=2, num_channels=channels[-1]),

                torch.nn.Conv2d(in_channels=channels[-1], out_channels=channels[-1], kernel_size=3, stride=1,
                                padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.GroupNorm(num_groups=2, num_channels=channels[-1]),
            ),
            torch.nn.Tanh()
        )
        channels = [32, 32, 16, 8, 4]
        self.decoder = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.GroupNorm(num_groups=2, num_channels=ch_out),

                torch.nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.GroupNorm(num_groups=2, num_channels=ch_out),
                torch.nn.UpsamplingBilinear2d(scale_factor=2)
            ) for ch_in, ch_out in zip(channels, channels[1:])],

            # last conv block built different
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=channels[-1], out_channels=channels[-1], kernel_size=3, stride=1,
                                padding=1),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.GroupNorm(num_groups=2, num_channels=channels[-1]),

                torch.nn.Conv2d(in_channels=channels[-1], out_channels=1, kernel_size=3, stride=1, padding=1),
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder.forward(x)
        z = z.view(-1, 32)
        # PCA
        z = z.view(-1, 32, 1, 1)
        out = self.decoder.forward(z) # TODO: (BATCH_SIZE, 1, 16, 16)
        return out


def test():
    model = AutoEncoder()
    summary(model, (1, 28, 28))
    dummy = torch.randn((32, 1, 28, 28))
    print(model.forward(dummy))


def main():
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

    model = AutoEncoder()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metrics = {}
    for stage in ['train', 'test']:
        for metric in ['loss']:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(EPOCHS):
        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            stage = 'train'
            torch.set_grad_enabled(True)
            if data_loader == data_loader_test:
                stage = 'test'
                torch.set_grad_enabled(False)
            for x, y, label in data_loader:
                # noisy image
                x = x.to(DEVICE)
                # real image
                y = y.to(DEVICE)
                # input noisy image, output denoised image
                y_prim = model.forward(x)
                loss = torch.mean((y - y_prim) ** 2)
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

        # draw losses
        c = 0
        for key, value in metrics.items():
            value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])
        # inference results, draw last batch noised and denoised image
        for i, j in enumerate([4, 5, 6, 16, 17, 18]):
            plt.subplot(4, 6, j)
            plt.title(f"class: {data_loader.dataset.data.classes[idx_label[i]]}")
            plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))

            plt.subplot(4, 6, j + 6)
            plt.imshow(np_y_prim[i][0].T, cmap=plt.get_cmap('Greys'))

        plt.tight_layout(pad=0.5)
        plt.show()

        # torch.save(model.state_dict(), './last-auto-encoder.pt')
    # input('quit?')


if __name__ == '__main__':
    main()

# TODO implement (MSA) Multiple Sequence Alignments, L1 loss
# TODO: use GroupNorm
# TODO: after training plot z using PCA
