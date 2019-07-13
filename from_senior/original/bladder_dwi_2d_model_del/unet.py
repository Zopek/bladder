import torch.nn as nn
import torch


def conv3x3_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


def maxpool2x2():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def upconv2x2(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def double_conv(in_channels, out_channels):
    return nn.Sequential(conv3x3_relu(in_channels, out_channels), conv3x3_relu(out_channels, out_channels))


class UNet(nn.Module):
    def __init__(self, num_class):
        super(UNet, self).__init__()
        self.en_0 = double_conv(3, 64)
        self.en_1 = double_conv(64, 128)
        self.en_2 = double_conv(128, 256)
        self.en_3 = double_conv(256, 512)
        self.en_4_0 = conv3x3_relu(512, 1024)
        self.en_4_1 = conv3x3_relu(1024, 1024)

        self.pool_0 = maxpool2x2()
        self.pool_1 = maxpool2x2()
        self.pool_2 = maxpool2x2()
        self.pool_3 = maxpool2x2()

        self.up_0 = upconv2x2(1024, 512)
        self.up_1 = upconv2x2(512, 256)
        self.up_2 = upconv2x2(256, 128)
        self.up_3 = upconv2x2(128, 64)

        self.de_0 = double_conv(1024, 512)
        self.de_1 = double_conv(512, 256)
        self.de_2 = double_conv(256, 128)
        self.de_3 = double_conv(128, 64)

        self.pred_0 = nn.Sequential(nn.Conv2d(1024, 1024, 1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True), nn.Conv2d(1024, num_class, 1))
        self.pred_1 = nn.Conv2d(64, num_class, 1)

    def forward(self, x):
        # x should be nx3x160x160
        x = self.en_0(x)
        skip_0 = x
        x = self.en_1(self.pool_0(x))
        skip_1 = x
        x = self.en_2(self.pool_1(x))
        skip_2 = x
        x = self.en_3(self.pool_2(x))
        skip_3 = x

        x = self.en_4_0(self.pool_3(x))
        pred_0 = self.pred_0(x)

        x = self.en_4_1(x)

        x = torch.cat([skip_3, self.up_0(x)], 1)
        x = self.de_0(x)
        x = torch.cat([skip_2, self.up_1(x)], 1)
        x = self.de_1(x)
        x = torch.cat([skip_1, self.up_2(x)], 1)
        x = self.de_2(x)
        x = torch.cat([skip_0, self.up_3(x)], 1)
        x = self.de_3(x)
        pred_1 = self.pred_1(x)
        return pred_0, pred_1
