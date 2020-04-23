import numpy as np

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


class Variable:
    def __init__(self, value: np.ndarray, grad: np.ndarray = None):
        # value computed in the forward pass
        self.value = value
        # derivative of circuit, computed in backward pass
        self.grad = grad
        if self.grad is None:
            self.grad = np.zeros_like(self.value)
        assert self.value.shape == self.grad.shape


# Applies a linear transformation to the incoming data: y = xW + b, coefficients W and b are differentiated
class LayerLinear:
    def __init__(self, in_features, out_features):
        # initialize coefficients with real numbers from 0..1
        self.w: Variable = Variable(np.random.uniform(size=(in_features, out_features)))
        self.b: Variable = Variable(np.random.uniform(size=out_features))
        self.x: Variable = None
        self.out: Variable = None

    def forward(self, x: Variable):
        self.x = x
        # matmul adds one more dimension, if one of arrays is one dimensional
        # (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html)
        self.out = Variable(np.matmul(self.x.value, self.w.value) + self.b.value)
        return self.out

    def backward(self):
        # w.grad = x.val x out.grad =>  (100,2,40) = (100,2,1) x (100,1,40)
        self.w.grad = np.matmul(np.expand_dims(self.x.value, axis=2), np.expand_dims(self.out.grad, axis=1))
        # w.grad => average to (2,40)
        self.w.grad = np.average(self.w.grad, axis=0)
        # b.grad => average from 1 x (100,40) to (1,40)
        self.b.grad = np.average(self.out.grad, axis=0)
        # out.grad need to be resized back with w.val, reverse process to linear transformation
        # x.grad = out.grad x w.val^T => (100,2) = (100, 40) x (40,2)
        self.x.grad = np.matmul(self.out.grad, np.transpose(self.w.value))

    def parameters(self):
        # return linear transformation coefficients for step update
        l = list()
        l.append(self.w)
        l.append(self.b)
        return l


# Applies a sigmoid function
class LayerSigmoid:
    def __init__(self):
        self.x: Variable = None
        self.out: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.out = Variable(1 / (1 + np.exp(-self.x.value)))
        return self.out

    def backward(self):
        # derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        self.x.grad = self.out.value * (1 - self.out.value) * self.out.grad

    def parameters(self):
        # return empty list, since no coefficients
        return list()


# Applies a ReLU(Rectified Linear Unit) function
# ReLU(x)=max(0,x)
class LayerReLU:
    def __init__(self):
        self.x: Variable = None
        self.out: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.out = Variable(self.x.value * (self.x.value > 0))
        return self.out

    def backward(self):
        # derivative of ReLU(x) = 1 if x is positive and 0 if not
        self.x.grad = (self.x.value > 0) * self.out.grad

    def parameters(self):
        # return empty list, since no coefficients
        return list()


# Applies a Tanh(Hyperbolic tangent) function
# Tanh(x)=tanh(x)= (e^x - e^(-x)) / (e^x + e^(-x))
class LayerTanh:
    def __init__(self):
        self.x: Variable = None
        self.out: Variable = None
        # positive and negative exponents
        self.e_p = None
        self.e_n = None

    def forward(self, x: Variable):
        self.x = x
        self.e_p = np.exp(self.x.value)
        self.e_n = np.exp(-self.x.value)
        tanh = (self.e_p - self.e_n) / (self.e_p + self.e_n)
        self.out = Variable(tanh)
        return self.out

    def backward(self):
        # derivative of Tanh(x) is square of hyperbolic secant
        # sech(x) = 2 / (e^x + e^(-x))
        # check https://www.math24.net/derivatives-hyperbolic-functions/
        sech = 2 / (self.e_p + self.e_n)
        self.x.grad = sech ** 2 * self.out.grad

    def parameters(self):
        # return empty list, since no coefficients
        return list()


# Applies the Softmax function to an n-dimensional input Tensor rescaling them
# so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
# TODO


# Applies a Mean Squared Error estimator
class LossMSE:
    def __init__(self):
        self.y_predicted: Variable = None
        self.y_correct: Variable = None
        self.out: Variable = None

    def forward(self, y_correct: Variable, y_predicted: Variable):
        self.y_predicted = y_predicted
        self.y_correct = y_correct
        self.out = np.mean((y_correct.value - y_predicted.value) ** 2)
        self.out = Variable(np.asarray([self.out]))
        return self.out

    def backward(self):
        self.y_predicted.grad = 2 * (1 / self.y_predicted.value.shape[0]) * (
                self.y_predicted.value - self.y_correct.value)


class FeedForwardNeuralNet:
    def __init__(self, layers: list):
        self.layers: list = layers

    def forward(self, x: Variable):
        out: Variable = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        parameters = list()
        for layer in self.layers:
            parameters.append(layer.parameters())
        # flatten
        flat_list = [item for sublist in parameters for item in sublist]
        return flat_list


class Optimiser:
    def __init__(self, parameters: list, lr):
        # list of variables
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            # to minimize loss function using gradient descent, need to go down with gradient
            p.value -= self.lr * p.grad


# coefficient of determination: https://en.wikipedia.org/wiki/Coefficient_of_determination
def r2_score(y_correct: np.ndarray, y_predicted: np.ndarray):
    mean_correct = np.mean(y_correct)

    # scikit r2_score uses 1 - ss_res / ss_tot
    # ss_res = np.sum((y_predicted - y_correct) ** 2)

    ss_reg = np.sum((y_predicted - mean_correct) ** 2)
    ss_tot = np.sum((y_correct - mean_correct) ** 2)
    return 1 - ss_reg / ss_tot


'''
Batch size better to be 2^n

Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

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

x_train, x_test = Variable(x[:train_size]), Variable(x[train_size:])
y_train, y_test = Variable(y[:train_size]), Variable(y[train_size:])

# Instantiate network

# layers = [LayerLinear(input_dim, 40),
#           LayerSigmoid(),
#           LayerLinear(40, 20),
#           LayerSigmoid(),
#           LayerLinear(20, output_dim)]

layers = [LayerLinear(input_dim, 40),
          LayerReLU(),
          LayerLinear(40, 20),
          LayerTanh(),
          LayerLinear(20, output_dim)]
model = FeedForwardNeuralNet(layers)

# Training
l_rate = 1e-3
loss = LossMSE()
optimiser = Optimiser(model.parameters(), lr=l_rate)

r2_scores = []
losses = []
for epoch in range(epochs):
    # collect metrics for each epoch (calculate average metrics of each batch)
    losses_epoch = []
    r2_scores_epoch = []

    # training data is divided by size of one batch and iterated
    for batch in range((train_size - batch_size) // batch_size):
        x_train_batch = Variable(x_train.value[(batch * batch_size): ((batch + 1) * batch_size):])
        y_train_batch = Variable(y_train.value[(batch * batch_size): ((batch + 1) * batch_size):])

        # forward
        y_predicted = model.forward(x_train_batch)

        # metrics
        losses_epoch.append(loss.forward(y_train_batch, y_predicted).value)
        r2_scores_epoch.append(r2_score(y_train.value, y_predicted.value))

        # backward
        loss.backward()
        model.backward()

        # update parameters
        optimiser.step()

    # Get mean loss and r2 score of epoch batches
    losses.append(np.mean(losses_epoch))
    r2_scores.append(np.mean(r2_scores_epoch))

    # Clear the current figure
    plt.clf()

    # Plot loss over epoch
    plt.subplot(2, 1, 1)
    plt.title('loss')
    plt.plot(np.arange(epoch + 1), np.array(losses), 'r-', label='loss')

    # Plot R2 score over epoch
    plt.subplot(2, 1, 2)
    plt.title('r2')
    plt.plot(np.arange(epoch + 1), np.array(r2_scores), 'r-', label='r2')

    plt.draw()
    plt.pause(0.1)
    # TODO: tkinter instantly closes when finishing to plot
