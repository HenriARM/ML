# PyTorch
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


# Test Non Linear regression with pytorch nn
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out


# generate the input data
x_vals = np.random.rand(500)
x_train = np.asarray(x_vals, dtype=np.float32).reshape(-1, 1)
y_correct = np.asarray([i ** 2 for i in x_vals], dtype=np.float32).reshape(-1, 1)

# instantiate network
input_dim = 1
output_dim = 1
# TODO: how hidden dimension size impacts on results (they proportionally till some size probably)
hidden_dim = 40
modelFF = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
criterionFF = nn.MSELoss()  # Mean Squared Loss
l_rate = 0.01

print("hello")
for param in modelFF.parameters():
    print(param)
print("hello")

optimiser = torch.optim.Adam(modelFF.parameters(), lr=l_rate)
epochs = 100

# Training
for epoch in range(epochs):
    epoch += 1
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_correct))

    # clear grads
    optimiser.zero_grad()

    # forward to get predicted values
    outputs = modelFF.forward(inputs)
    loss = criterionFF(outputs, labels)
    loss.backward()  # back props
    optimiser.step()  # update the parameters
    if (epoch + 1) % 500 == 0:  # Logging
        print('Epoch [%d/%d], Loss: %.4f'
              % (epoch + 1, epochs, loss.item()))
        print('Final - epoch {}, loss {}'.format(epoch, loss.item()))

# Test training
predicted = modelFF.forward(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_correct, 'go', label='from data', alpha=0.2)
plt.plot(x_train, predicted, 'b*', label='prediction', alpha=0.2)
plt.legend()
plt.show()
