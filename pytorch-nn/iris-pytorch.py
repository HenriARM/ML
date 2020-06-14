import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import matplotlib

# scikit only used to load Iris dataset
from sklearn.datasets import load_iris

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Applies a Cross-Entropy Loss
class LossCrossEntropy:
    def __init__(self):
        self.y_predicted = None
        self.y_correct = None

    def forward(self, y_predicted, y_correct):
        self.y_predicted = y_predicted
        self.y_correct = y_correct
        # since correct output is a one-hot encoded vector, where all values == 0.0, except one which == 1.0
        # we can simplify sum of probability multiplication to
        return torch.sum(-torch.log(self.y_predicted[self.y_correct == 1.0]))


class IrisDataset(Dataset):
    def __init__(self, data: np.ndarray, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, id):
        sample = self.data[id]
        one_hot = np.zeros(OUTPUT_DIM)
        one_hot[int(sample[4])] = 1.0
        sample = {'input': sample[:4], 'output': one_hot}
        if self.transform:
            sample = self.transform(sample)
        return sample


BATCH_SIZE = 15
EPOCHS = 10
INPUT_DIM = 4
OUTPUT_DIM = 3


def main():
    # Load iris dataset with total of 150 samples
    x, y = load_iris(return_X_y=True)
    y = np.reshape(y, (150, 1))
    # concatenate both x and y to shuffle and split
    data = np.concatenate((x, y), axis=1)

    dataset = IrisDataset(data)
    dataset_train, dataset_test = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])

    # https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/dataloader.html
    data_loader_train = DataLoader(
        dataset_train,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    data_loader_test = DataLoader(
        dataset_test,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = nn.Sequential(nn.Linear(INPUT_DIM, 48, bias=True),
                          nn.ReLU(),
                          nn.Linear(48, 24, bias=True),
                          nn.Tanh(),
                          nn.Linear(24, OUTPUT_DIM, bias=True),
                          nn.Softmax(dim=1))

    # Training properties
    l_rate = 1e-3
    criterion = LossCrossEntropy()
    optimiser = torch.optim.Adam(model.parameters(), lr=l_rate)

    losses = []
    for epoch in range(EPOCHS):
        print("\nepoch = ", epoch)

        # collect metrics for each epoch (calculate average metrics of each batch)
        losses_epoch = []
        for data_loader in [data_loader_train, data_loader_test]:
            if data_loader == data_loader_train:
                print("\n\ttraining:")
                model = model.train()
                torch.set_grad_enabled(True)
            else:
                print("\n\ttesting:")
                model = model.eval()
                torch.set_grad_enabled(False)
            for i_batch, batch in enumerate(data_loader):
                print("\t\tbatch = ", i_batch)

                # clear grads
                optimiser.zero_grad()

                # tensors
                x_train_batch = batch["input"]
                y_train_batch = batch["output"]

                # forward
                y_predicted_batch = model.forward(x_train_batch.float())

                if data_loader == data_loader_train:
                    # loss backward
                    loss = criterion.forward(y_predicted_batch, y_train_batch)
                    loss.backward()

                    # update parameters
                    optimiser.step()

                    # metrics
                    losses_epoch.append(loss.detach().numpy())

        # Get mean loss and metrics of epoch batches
        losses.append(np.mean(losses_epoch))

        # Clear the current figure
        plt.clf()

        # Plot loss over epoch
        plt.subplot(1, 1, 1)
        plt.title('loss')
        plt.plot(np.arange(epoch + 1), np.array(losses), 'r-', label='loss')

        plt.draw()
        plt.pause(0.1)

    # wait for keyboard press to close tkAgg
    while True:
        if plt.waitforbuttonpress():
            exit(0)


if __name__ == '__main__':
    main()
