# PyTorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Applies a Mean Squared Error Loss
class LossMSE:
    def __init__(self):
        self.y_predicted = None
        self.y_correct = None

    def forward(self, y_predicted, y_correct):
        self.y_predicted = y_predicted
        self.y_correct = y_correct
        return torch.mean((self.y_correct - self.y_predicted) ** 2)


def main():
    # Generate the input datasets
    batch_size = 32
    epochs = 10
    input_dim = 2
    output_dim = 1
    train_size = batch_size * 80
    test_size = batch_size * 20

    x = np.random.random((train_size + test_size, input_dim))
    y = np.asarray([i[0] ** 2 + i[1] for i in x], dtype=np.float32).reshape(-1, 1)

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = nn.Sequential(nn.Linear(input_dim, 40, bias=True),
                          nn.Sigmoid(),
                          nn.Linear(40, 20, bias=True),
                          nn.Sigmoid(),
                          nn.Linear(20, output_dim, bias=True))

    # Training
    l_rate = 1e-3
    loss = LossMSE()
    optimiser = torch.optim.Adam(model.parameters(), lr=l_rate)

    # Training
    for epoch in range(epochs):
        # collect metrics for each epoch (calculate average metrics of each batch)
        losses_epoch = []
        r2_scores_epoch = []

        # training datasets is divided by size of one batch and iterated
        for batch in range((train_size - batch_size) // batch_size):
            # clear grads
            optimiser.zero_grad()

            # create tensors
            x_train_batch = torch.from_numpy(x_train[(batch * batch_size): ((batch + 1) * batch_size):])
            y_train_batch = torch.from_numpy(y_train[(batch * batch_size): ((batch + 1) * batch_size):])

            # forward
            y_predicted_batch = model.forward(x_train_batch.float())

            # loss backward
            output = loss.forward(y_predicted_batch, y_train_batch)
            output.backward()

            # update parameters
            optimiser.step()

            # metrics
            # losses_epoch.append(loss.forward(y_train_batch, y_predicted).value)
            # r2_scores_epoch.append(r2_score(y_train.value, y_predicted.value))

    # Test
    y_predicted = model.forward(torch.from_numpy(x_test).float()).data.numpy()
    plt.plot(x_test, y_test, 'go', label='from datasets', alpha=0.2)
    plt.plot(x_test, y_predicted, 'b*', label='prediction', alpha=0.2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
