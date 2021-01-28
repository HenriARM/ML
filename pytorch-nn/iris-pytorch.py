import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tensorboardX import SummaryWriter

# to run one experiment $ tensorboard --logdir ./runs/exp1
# to run all experiments $ tensorboard --logdir ./runs
tb = SummaryWriter(log_dir='runs/exp1')

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

    def __getitem__(self, idx):
        sample = self.data[idx]
        one_hot = np.zeros(OUTPUT_DIM)
        # label is on the latest column of input array
        one_hot[int(sample[OUTPUT_DIM + 1])] = 1.0
        # samples last column not used in x
        sample = sample[:4], one_hot, idx
        # if self.transform:
        #     sample = self.transform(sample)
        # TODO: how to convert x and y to float32 tensors.
        #  Dataset by default at the end converts all batch to tensor float64
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
    train_loader = DataLoader(
        dataset_train,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
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
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    meters = {
        "train_loss": AverageValueMeter(),
        "test_loss": AverageValueMeter(),

        "train_acc": ClassErrorMeter(accuracy=True),
        "test_acc": ClassErrorMeter(accuracy=True),

        "train_f1": MetricAccuracyClassification(),
        "test_f1": MetricAccuracyClassification()
    }

    for epoch in range(EPOCHS):
        # print("\nepoch = ", epoch)

        # reset meters to default settings before each epoch
        for k in meters.keys():
            meters[k].reset()

        for loader in [train_loader, test_loader]:
            if loader == train_loader:
                # print("\n\ttraining:")
                meter_prefix = "train"
                model = model.train()
                torch.set_grad_enabled(True)
            else:
                # print("\n\ttesting:")
                meter_prefix = "test"
                # automatically turns off some modules (like DropOut), which are not used during testing
                model = model.eval()
                torch.set_grad_enabled(False)

            for x, y, idx in loader:
                # print("\t\tbatch = ", idx)

                # clear grads
                optimizer.zero_grad()

                # forward
                y_prim = model.forward(x.float())
                loss = criterion.forward(y_prim, y)

                # update parameters when training
                if loader == train_loader:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # update metrics
                meters[f'{meter_prefix}_loss'].add(loss.detach().numpy())

                # TODO: acc curve?
                # # convert from one hot encoding to flower classes
                # tg = np.argmax(y, axis=1).numpy()
                # (y_prim.detach().numpy(), target=tg)

                # TODO: f1

            tb.add_scalars(
                main_tag='learning_curve',
                tag_scalar_dict={
                    f'{meter_prefix}_loss': meters[f'{meter_prefix}_loss'].value()[0]
                },
                global_step=epoch
            )

    tb.close()


if __name__ == '__main__':
    main()
    # writer = SummaryWriter(log_dir='runs/exp1')
    # x = range(10)
    # for i in x:
    #     writer.add_scalar('y=3x', i * 3, i)
    # writer.close()
