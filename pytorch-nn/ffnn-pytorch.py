# PyTorch
import torch
import torch.nn as nn
import numpy as np
# from torch.autograd import Variable
import matplotlib.pyplot as plt


# TODO: how can i send as a param pytorch layer classes (e.x. like a list)
# nn.Module is a Base class for all neural network modules
class FeedForwardNeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_dim, 40, bias=True)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(40, 20, bias=True)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(20, output_dim, bias=True)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        return self.l5(out)


# Generate the input data

batch_size = 32
epochs = 10
input_dim = 2
output_dim = 1
train_size = batch_size * 80
test_size = batch_size * 20

x = np.random.random((train_size + test_size, input_dim))
# y = x_1^2 + x_2
y = np.asarray([i[0] ** 2 + i[1] for i in x], dtype=np.float32).reshape(-1, 1)

x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = FeedForwardNeuralNet(input_dim, output_dim)

# Training
l_rate = 1e-3
loss = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=l_rate)

# Training
for epoch in range(epochs):
    # collect metrics for each epoch (calculate average metrics of each batch)
    losses_epoch = []
    r2_scores_epoch = []

    # training data is divided by size of one batch and iterated
    for batch in range((train_size - batch_size) // batch_size):
        # clear grads
        optimiser.zero_grad()

        # TODO: we can send tensors, dont need Vairable structure
        # x_train_batch = Variable(torch.from_numpy(x_train[(batch * batch_size): ((batch + 1) * batch_size):]))
        # y_train_batch = Variable(torch.from_numpy(y_train[(batch * batch_size): ((batch + 1) * batch_size):]))

        # create tensors
        x_train_batch = torch.from_numpy(x_train[(batch * batch_size): ((batch + 1) * batch_size):])
        y_train_batch = torch.from_numpy(y_train[(batch * batch_size): ((batch + 1) * batch_size):])

        # forward
        y_predicted = model.forward(x_train_batch.float())

        # loss backward
        output = loss(y_predicted, y_train_batch)
        output.backward()

        # update parameters
        optimiser.step()

        # metrics
        # TODO: add loss value
        # losses_epoch.append(loss.forward(y_train_batch, y_predicted).value)
        # r2_scores_epoch.append(r2_score(y_train.value, y_predicted.value))

# Test
y_predicted = model.forward(torch.from_numpy(x_test).float()).data.numpy()
plt.plot(x_test, y_test, 'go', label='from data', alpha=0.2)
plt.plot(x_test, y_predicted, 'b*', label='prediction', alpha=0.2)
plt.legend()
plt.show()
