# PyTorch
import torch
import torch.nn as nn

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


def main():
    # Load iris dataset with total of 150 samples
    x, y = load_iris(return_X_y=True)
    y = np.reshape(y, (150, 1))
    # concatenate both x and y to shuffle
    data = np.concatenate((x, y), axis=1)
    np.random.shuffle(data)

    # input size is 150
    batch_size = 15
    epochs = 10
    input_dim = 4
    # on output we get probability of each flower
    output_dim = 3
    train_size = batch_size * 8
    test_size = batch_size * 2

    # instead of storing flower labels, convert to one hot encoding, last data column
    y = data[:, 4]
    one_hot = np.zeros((y.size, 3))
    # arange() creates array with integer indices
    one_hot[np.arange(y.size), y.astype(int)] = 1.0
    y = one_hot

    # split into train and test data
    # remove y as a last column
    x = np.delete(data, 4, axis=1)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = nn.Sequential(nn.Linear(input_dim, 48, bias=True),
                          nn.ReLU(),
                          nn.Linear(48, 24, bias=True),
                          nn.Tanh(),
                          nn.Linear(24, output_dim, bias=True),
                          nn.Softmax(dim=1))

    # Training
    l_rate = 1e-3
    criterion = LossCrossEntropy()
    optimiser = torch.optim.Adam(model.parameters(), lr=l_rate)

    # Training
    losses = []
    for epoch in range(epochs):
        # collect metrics for each epoch (calculate average metrics of each batch)
        losses_epoch = []

        # training data is divided by size of one batch and iterated
        for batch in range((train_size - batch_size) // batch_size):
            # clear grads
            optimiser.zero_grad()

            # create tensors
            x_train_batch = torch.from_numpy(x_train[(batch * batch_size): ((batch + 1) * batch_size):])
            y_train_batch = torch.from_numpy(y_train[(batch * batch_size): ((batch + 1) * batch_size):])

            # forward
            y_predicted_batch = model.forward(x_train_batch.float())

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
            break


if __name__ == '__main__':
    main()
