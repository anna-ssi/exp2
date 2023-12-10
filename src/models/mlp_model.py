import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, input_size=64, output_size=64):
        super(EncoderDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size * 2)
        self.fc2 = nn.Linear(output_size *2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x