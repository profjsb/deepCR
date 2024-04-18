from torch import sigmoid
from deepCR.parts_batchnorm import *


class UNet2SigmoidBatchNorm(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32):
        super(type(self), self).__init__()
        self.inc = inconv(n_channels, hidden)
        self.down1 = down(hidden, hidden * 2)
        self.up8 = up(hidden * 2, hidden)
        self.outc = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up8(x2, x1)
        x = self.outc(x)
        return sigmoid(x)


class UNet2BatchNorm(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32):
        super(type(self), self).__init__()
        self.inc = inconv(n_channels, hidden)
        self.down1 = down(hidden, hidden * 2)
        self.up8 = up(hidden * 2, hidden)
        self.outc = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up8(x2, x1)
        x = self.outc(x)
        return x


class UNet3BatchNorm(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32):
        super(type(self), self).__init__()
        self.inc = inconv(n_channels, hidden)
        self.down1 = down(hidden, hidden * 2)
        self.down2 = down(hidden * 2, hidden * 4)
        self.up7 = up(hidden * 4, hidden * 2)
        self.up8 = up(hidden * 2, hidden)
        self.outc = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up7(x3, x2)
        x = self.up8(x, x1)
        x = self.outc(x)
        return x


class WrappedModel(nn.Module):
    def __init__(self, network):
        super(type(self), self).__init__()
        self.module = network

    def forward(self, *x):
        return self.module(*x)