import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import tensorboardX

tb = tensorboardX.SummaryWriter(log_dir='./iris')

from torchnet.meter import AverageValueMeter, ClassErrorMeter
from metrics_utils import MetricAccuracyClassification

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
        # label is on the latest column of input array
        one_hot[int(sample[4])] = 1.0
        # TODO:
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

    meters = {
        "train_loss": AverageValueMeter(),
        "test_loss": AverageValueMeter(),

        "train_acc": ClassErrorMeter(accuracy=True),
        "test_acc": ClassErrorMeter(accuracy=True)
    }

    # TODO: convert to meters
    # train_losses = []
    # test_losses = []

    for epoch in range(EPOCHS):
        print("\nepoch = ", epoch)

        # reset meters to default settings before each epoch
        for k in meters.keys():
            meters[k].reset()

        for data_loader in [data_loader_train, data_loader_test]:
            if data_loader == data_loader_train:
                print("\n\ttraining:")
                meter_prefix = "train"
                model = model.train()
                torch.set_grad_enabled(True)
            else:
                print("\n\ttesting:")
                meter_prefix = "test"
                # automatically turns off some modules (like DropOut), which are not used during testing
                model = model.eval()
                torch.set_grad_enabled(False)

            meter_batch_loss = AverageValueMeter()
            # meter_batch_F1 = MetricAccuracyClassification()

            for i_batch, batch in enumerate(data_loader):
                print("\t\tbatch = ", i_batch)

                # clear grads
                optimiser.zero_grad()

                # tensors
                # TODO:
                # you should get in loop already X_train, Y_train, to do that,
                # you need to send back tuple for data_Loader
                x_batch = batch["input"]
                y_batch = batch["output"]

                # forward
                y_predicted_batch = model.forward(x_batch.float())
                loss = criterion.forward(y_predicted_batch, y_batch)

                # update parameters when training
                if data_loader == data_loader_train:
                    loss.backward()
                    optimiser.step()

                # update metrics
                meter_batch_loss.add(loss.detach().numpy())
                # TODO: assertion error
                meters[f'{meter_prefix}_acc'].add(y_predicted_batch.detach().numpy(), y_batch.detach().numpy())
                # TODO: do F1 score, store two variables both for train and test. Add fp, tp during each batch:
                # meterF1.fp += 1

            meters[f'{meter_prefix}_loss'].add(meter_batch_loss.value()[0])

            tb.add_scalar(
                tag=f'{meter_prefix}_loss',
                scalar_value=meter_batch_loss.value()[0],
                global_step=epoch
            )

            # TODO: temp code
            # temp_loss = meter_batch_loss.value()[0]
            # if data_loader == data_loader_train:
            #     train_losses.append(temp_loss)
            # else:
            #     test_losses.append(temp_loss)


    # TODO: how to plot each epoch, not gather as array and plot at the end?
    # fig, ax = plt.subplots()
    # ax.plot(train_losses, test_losses)
    #
    # tb.add_figure(
    #     tag='train_test_loss',
    #     figure=fig,
    #     # global_step=epoch
    # )

    tb.close()

    # TODO: print accuracoies
    # print('Train Acc:' + meters['train_acc'].value())
    # print('Test Acc:' + meters['test_acc'].value())


if __name__ == '__main__':
    main()
