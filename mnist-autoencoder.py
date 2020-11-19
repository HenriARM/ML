import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.autograd import Variable
from torch.optim import Adam


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels=6, in_channels=3, kernel_size=5),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# # 32 x 32 - cifar
# # 28 x 28 - mnist


def main():
    # TODO: add normalizations=
    # train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root='mnist-dataset/', train=True, download=True, transform=transform)

    BATCH_SIZE = 256
    EPOCHS = 2
    lr = 1e-3

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # test_loader

    # #  use gpu if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # load it to the specified device, either gpu or cpu
    # model = AE(input_shape=784).to(device)

    model = AE().cpu()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(EPOCHS):
        for image, _ in train_loader:
            # label is not used  in AE
            # image = Variable(image).cpu()

            # image = image.view(-1, 784) #.to(device)

            output = model.forward(image)
            # TODO: convert output image too
            loss = criterion.forward(output, image)

            # reset gradients
            optimizer.zero_grad()

            criterion.backward()
            optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, EPOCHS, criterion.data()))


if __name__ == '__main__':
    main()
