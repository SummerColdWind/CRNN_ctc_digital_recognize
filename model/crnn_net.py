import torch
from torch.nn import Conv2d, ReLU, MaxPool2d, LSTM, Linear

from typing import Tuple

class Net(torch.nn.Module):
    def __init__(
            self,
            image_shape: Tuple[int, int],
            classes_num: int
    ):
        super().__init__()
        lstm_input_size = (image_shape[0] // 2 // 2) * 128
        self.conv_layer = torch.nn.Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_input_size,
            num_layers=1,
            bidirectional=True
        )
        self.fc = Linear(
            in_features=lstm_input_size * 2,
            out_features=classes_num
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.shape[0], x.shape[1], -1)
        x, _ = self.lstm(x)
        shape = x.shape
        x = x.view(-1, shape[2])
        x = self.fc(x)
        x = x.view(shape[0], shape[1], -1)
        return x
