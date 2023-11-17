import torch
import torch.nn as nn

# TODO: look at the dimensions


class ConvNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ZeroPad2d((15, 15, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=20,
                      kernel_size=(1, 31), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40,
                      kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=80,
                      kernel_size=(1, 21), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))

        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))

        self.layer4 = nn.Sequential(
            # nn.ZeroPad2d((15,15,0,0)),
            nn.Conv2d(in_channels=80, out_channels=160,
                      kernel_size=(1, 11), stride=(1, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))

        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160,
                      kernel_size=(7, 1), stride=(7, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU())

        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))

        self.linear1 = nn.Sequential(
            nn.Linear(160*4, num_classes),
            nn.LogSoftmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool2(out)
        out = self.layer4(out)
        out = self.pool3(out)
        out = self.layer5(out)
        out = self.pool4(out)

        out = torch.flatten(out, start_dim=1)
        out = self.linear1(out)
        return out
