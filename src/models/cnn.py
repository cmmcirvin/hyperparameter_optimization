import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Configurable CNN model for MNIST1d."""

    def __init__(self, num_classes, p_dropout=0.2, width=32, height=2, fully_connected=2):
        nn.Module.__init__(self)

        self._num_classes = num_classes

        self._dropout = nn.Dropout(p_dropout)

        self._conv_layers = nn.ModuleList()
        self._bn_layers = nn.ModuleList()

        self._conv_layers.append(nn.Conv1d(1, width, 3, 1))
        self._bn_layers.append(nn.BatchNorm1d(width))
        for _ in range(height):
            self._conv_layers.append(nn.Conv1d(width, width, 3, 1))
            self._bn_layers.append(nn.BatchNorm1d(width))

        self._fc_layers = nn.ModuleList()
        for _ in range(fully_connected - 1):
            self._fc_layers.append(nn.LazyLinear(width * width))
        self._fc_layers.append(nn.LazyLinear(num_classes))

    def forward(self, x):

        for i in range(len(self._conv_layers)):
            x = F.relu(self._bn_layers[i](self._conv_layers[i](x)))

        x = x.flatten(1)

        for i in range(len(self._fc_layers)):
            x = self._dropout(x)
            x = F.relu(self._fc_layers[i](x))

        return x
