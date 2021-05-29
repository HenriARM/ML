import numpy as np

# scikit only used to load Iris dataset
from sklearn.datasets import load_iris

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


# Applies a linear transformation to the incoming datasets: y = xW + b, coefficients W and b are differentiated
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


# Applies a sigmoid function y(x) = 1 / (1 + e^(-x))
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
        self.out = Variable(np.maximum(self.x.value, np.zeros_like(self.x.value)))
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
# shift in forward() is used to minimize calculations <=> dividing all e^x to constant e^D, to calculate only e^x-D
class LayerSoftmax:
    def __init__(self):
        self.x: Variable = None
        self.out: Variable = None
        self.e_x = None
        self.jacobian = None

    def forward(self, x: Variable):
        self.x = x
        # get max elem of each line
        self.e_x = np.exp(self.x.value - np.max(self.x.value, axis=1, keepdims=True))
        # sum each line <=> second dimension
        self.out = Variable(self.e_x / np.sum(self.e_x, axis=1, keepdims=True))
        return self.out

    def backward(self):
        # create Jacobian matrix NxN with softmax derivatives, e.x. (100, 2, 2),
        #   where input_size = 100 and input_dim = 2
        self.jacobian = np.zeros((self.x.grad.shape[0], self.x.grad.shape[1], self.x.grad.shape[1]))
        # D_i S_j = S_i (1 - S_i) if i = j | - S_j * S_i if i != j;
        # check https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # jacobian[k,i,j]
        for k in range(self.jacobian.shape[0]):
            for i in range(self.jacobian.shape[1]):
                for j in range(self.jacobian.shape[2]):
                    if i == j:
                        self.jacobian[k, i, j] = self.out.value[k, i] * (1 - self.out.value[k, i])
                    else:
                        self.jacobian[k, i, j] = -self.out.value[k, i] * self.out.value[k, j]

        # (100, 1, 2) = (100, 1, 2) x (100, 2, 2)
        self.x.grad = np.matmul(np.expand_dims(self.out.grad, axis=1), self.jacobian)
        # average of second axis, to get (100,2)
        self.x.grad = np.average(self.x.grad, axis=1)

    # def backward(self):
    #     for idx in range(self.out.value.shape[0]):
    #         J = np.zeros((self.out.value.shape[1], self.out.value.shape[1]))
    #         for i in range(self.out.value.shape[1]):
    #             for j in range(self.out.value.shape[1]):
    #                 if i == j:
    #                     J[i, j] = self.out.value[idx][i] * (1 - self.out.value[idx][j])
    #                 else:
    #                     J[i, j] = -self.out.value[idx][i] * self.out.value[idx][j]
    #
    #         self.x.grad[idx] = np.matmul(J, self.out.grad[idx])

    def parameters(self):
        # return empty list, since no coefficients
        return list()


# Applies a Mean Squared Error Loss
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


# Applies a Cross-Entropy Loss
class LossCrossEntropy:
    def __init__(self):
        self.y_predicted: Variable = None
        self.y_correct: Variable = None
        self.out: Variable = None

    def forward(self, y_correct: Variable, y_predicted: Variable):
        self.y_predicted = y_predicted

        # tackle with log(0) and with null pointer exception when calculating grad
        epsilon = 1e-6
        self.y_predicted.value += epsilon

        self.y_correct = y_correct
        # output of cross entropy is sum of logarithms of predicted values  multiplied by probability
        self.out = -np.sum(y_correct.value * np.log(self.y_predicted.value))
        self.out = Variable(np.asarray(self.out))
        return self.out

    def backward(self):
        self.y_predicted.grad = self.y_correct.value / self.y_predicted.value


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
            # to minimize loss function using gradient descent
            # updating weights using addition is used for cross entropy loss (for mse should subtract grad)
            p.value += self.lr * p.grad


# coefficient of determination: https://en.wikipedia.org/wiki/Coefficient_of_determination
def r2_score(y_correct: np.ndarray, y_predicted: np.ndarray):
    mean_correct = np.mean(y_correct)

    # scikit r2_score uses 1 - ss_res / ss_tot
    # ss_res = np.sum((y_predicted - y_correct) ** 2)

    ss_reg = np.sum((y_predicted - mean_correct) ** 2)
    ss_tot = np.sum((y_correct - mean_correct) ** 2)
    return 1 - ss_reg / ss_tot


# accuracy = correct_samples / all_samples
def accuracy(y_correct: np.ndarray, y_predicted: np.ndarray):
    # sample is correct if correct label had maximum probability
    # will do in trickier way: get index array of correct labels and predicted labels and compare
    acc = 0
    a1 = np.argmax(y_predicted, axis=1)
    a2 = np.argmax(y_correct, axis=1)
    for i in range(a1.shape[0]):
        if a1[i] == a2[i]:
            acc += 1
    return acc / y_predicted.shape[0]


# returns macro-averaged F1-score, mean of F1-score per class
# there is also weighted-average F1-score, when amount of samples of different classes is various
# there is also micro-averaged F1-score which doesn't make sense since micro-recall is equal to micro-precision and
# micro-averaged F1-score is equal to accuracy
# https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
def f1_score(y_correct: np.ndarray, y_predicted: np.ndarray):
    false_positive = 0
    false_negative = 0

    predicted = np.argmax(y_predicted, axis=1)
    correct = np.argmax(y_correct, axis=1)

    # NxN matrix (N is different class amount) as in tutorial where column used for correct datasets and rows for predicted
    table = np.zeros((y_correct.shape[1], y_predicted.shape[1]))
    for i in range(predicted.shape[0]):
        table[predicted[i], correct[i]] += 1

    # for each class calculate f1 score
    f1_scores = []
    sum_rows = np.sum(table, axis=0)
    sum_columns = np.sum(table, axis=1)
    for i in range(y_predicted.shape[1]):
        # true positive lies on diagonal
        t_p = table[i, i]
        # true_positive + false_positive is a table's row
        t_p_and_f_p = sum_rows[i]
        # true_positive + false_negative is a table's column
        t_p_and_f_n = sum_columns[i]

        precision = 0
        recall = 0

        # at start there will be all samples predicted to one class, that's why check for null exception
        if t_p_and_f_p != 0:
            # number of correct positive results divided by the number of all positive results
            # returned by the classifier
            precision = t_p / t_p_and_f_p

        if t_p_and_f_n != 0:
            # number of correct positive results divided by all samples
            # that should have been identified as positive
            recall = t_p / t_p_and_f_n

        # add f1_score to list
        if precision != 0 or recall != 0:
            f1_scores.append(2 * (precision * recall) / (precision + recall))

    return np.mean(f1_scores)


'''
Batch size better to be 2^n

Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''


# TODO: add Cross Validation
def main():
    # Load iris dataset with total of 150 samples
    x, y = load_iris(return_X_y=True)
    y = np.reshape(y, (150, 1))
    # concatenate both x and y to cross validate
    data = np.concatenate((x, y), axis=1)
    np.random.shuffle(data)

    # input size = 150
    batch_size = 5
    epochs = 50
    input_dim = 4
    # on output we get probability of each flower
    output_dim = 3
    train_size = batch_size * 8
    test_size = batch_size * 2

    # instead of storing flower labels, convert to one hot encoding
    y = data[:, 4]
    one_hot = np.zeros((y.size, 3))
    # arange() creates array with integer indices
    one_hot[np.arange(y.size), y.astype(int)] = 1.0
    y = one_hot

    # split into train and test datasets
    # remove y as a last column
    x = np.delete(data, 4, axis=1)
    x_train, x_test = Variable(x[:train_size]), Variable(x[train_size:])
    y_train, y_test = Variable(y[:train_size]), Variable(y[train_size:])

    # Instantiate network
    layers = [LayerLinear(input_dim, 8),
              # LayerSigmoid(),
              LayerReLU(),
              LayerLinear(8, 8),
              LayerReLU(),
              # LayerTanh(),
              # LayerSigmoid(),
              LayerLinear(8, output_dim),
              LayerSoftmax()]
    model = FeedForwardNeuralNet(layers)

    # Training
    l_rate = 1e-2
    criterion = LossCrossEntropy()
    optimiser = Optimiser(model.parameters(), lr=l_rate)

    losses = []
    accuracies = []
    r2_scores = []
    f1_scores = []
    for epoch in range(epochs):

        # collect metrics for each epoch (calculate average metrics of each batch)
        losses_epoch = []
        accuracies_epoch = []
        r2_scores_epoch = []
        f1_scores_epoch = []

        # training datasets is divided by size of one batch and iterated
        for batch in range((train_size - batch_size) // batch_size):
            x_train_batch = Variable(x_train.value[(batch * batch_size): ((batch + 1) * batch_size):])
            y_train_batch = Variable(y_train.value[(batch * batch_size): ((batch + 1) * batch_size):])

            # forward
            y_predicted = model.forward(x_train_batch)

            # loss and metrics
            loss = criterion.forward(y_train_batch, y_predicted)
            losses_epoch.append(loss.value)
            accuracies_epoch.append(accuracy(y_train_batch.value, y_predicted.value))
            r2_scores_epoch.append(r2_score(y_train_batch.value, y_predicted.value))
            f1_scores_epoch.append(f1_score(y_train_batch.value, y_predicted.value))

            # backward
            criterion.backward()
            model.backward()

            # update parameters
            optimiser.step()

            print(f'loss: {loss.value}')

        # Get mean loss and metrics of epoch batches
        losses.append(np.mean(losses_epoch))
        accuracies.append(np.mean(accuracies_epoch))
        r2_scores.append(np.mean(r2_scores_epoch))
        f1_scores.append(np.mean(f1_scores_epoch))

        plt.clf()
        plt.title('loss')
        plt.plot(np.arange(epoch + 1), np.array(losses), 'r-', label='train_loss')
        plt.draw()
        plt.pause(0.1)

        # # Clear the current figure
        # plt.clf()
        #
        # # Plot loss over epoch
        # plt.subplot(4, 1, 1)
        # plt.title('loss')
        # plt.plot(np.arange(epoch + 1), np.array(losses), 'r-', label='loss')
        #
        # # Plot R2 score over epoch
        # plt.subplot(4, 1, 2)
        # plt.title('r2')
        # plt.plot(np.arange(epoch + 1), np.array(r2_scores), 'r-', label='r2')
        #
        # # Plot F1 score over epoch
        # plt.subplot(4, 1, 3)
        # plt.title('f1')
        # plt.plot(np.arange(epoch + 1), np.array(f1_scores), 'r-', label='f1')
        #
        # # Plot accuracy over epoch
        # plt.subplot(4, 1, 4)
        # plt.title('accuracy')
        # plt.plot(np.arange(epoch + 1), np.array(accuracies), 'r-', label='accuracy')
        #
        # plt.draw()
        # plt.pause(0.1)

    # wait for keyboard press to close tkAgg
    while True:
        if plt.waitforbuttonpress():
            break

    # Test
    y_predicted = model.forward(x_test)

    # Metrics
    print("On Test datasets, results are following:\n")
    print("loss = ", np.mean(criterion.forward(y_test, y_predicted).value))
    print("r2 = ", r2_score(y_test.value, y_predicted.value))
    print("f1 = ", f1_score(y_test.value, y_predicted.value))
    print("acc = ", accuracy(y_test.value, y_predicted.value))


if __name__ == '__main__':
    main()
