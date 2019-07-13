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


def pred_layer(in_channels, out_channels, dropout_prob):
    if dropout_prob != 0.0:
        return nn.Sequential(nn.Dropout2d(p=dropout_prob, inplace=False),
                             nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.ReLU(inplace=True),
                             nn.Dropout2d(p=dropout_prob, inplace=True),
                             nn.Conv2d(in_channels, out_channels, kernel_size=1))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels, out_channels, kernel_size=1))


class EncodeAndPredictLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, use_pooling, concat_pred, share, dropout_prob):
        print('encodelayer:', in_channels, out_channels, num_class, use_pooling, concat_pred, share, dropout_prob)
        super(EncodeAndPredictLayer, self).__init__()
        self.use_pooling = use_pooling
        self.share = share
        if use_pooling:
            self.pool = maxpool2x2()
        self.concat_pred = concat_pred
        self.conv_0 = conv3x3_relu(in_channels, out_channels)
        self.pred = pred_layer(out_channels, num_class, dropout_prob)
        self.conv_1 = conv3x3_relu(out_channels + num_class * int(concat_pred), out_channels)
        if not share:
            if use_pooling:
                self.pool_strong = maxpool2x2()
            self.conv_0_strong = conv3x3_relu(in_channels, out_channels)
            self.conv_1_strong = conv3x3_relu(out_channels, out_channels)

    def forward(self, x_weak, x_strong):
        if self.use_pooling:
            x_weak = self.pool(x_weak)
        x_weak = self.conv_0(x_weak)
        pred = self.pred(x_weak)
        if self.concat_pred:
            x_weak = torch.cat([x_weak, pred], 1)
        x_weak = self.conv_1(x_weak)

        if self.share:
            x_strong = x_weak
        else:
            if self.use_pooling:
                x_strong = self.pool_strong(x_strong)
            x_strong = self.conv_0_strong(x_strong)
            x_strong = self.conv_1_strong(x_strong)

        return x_weak, x_strong, pred


class UNet(nn.Module):
    def __init__(self, num_class, concat_pred_list, num_shared_encoders, dropout_prob_list):
        super(UNet, self).__init__()

        share_list = [i < num_shared_encoders for i in range(5)]
        self.en_0 = EncodeAndPredictLayer(3, 64, num_class, False, concat_pred_list[0], share_list[0], dropout_prob_list[0])
        self.en_1 = EncodeAndPredictLayer(64, 128, num_class, True, concat_pred_list[1], share_list[1], dropout_prob_list[1])
        self.en_2 = EncodeAndPredictLayer(128, 256, num_class, True, concat_pred_list[2], share_list[2], dropout_prob_list[2])
        self.en_3 = EncodeAndPredictLayer(256, 512, num_class, True, concat_pred_list[3], share_list[3], dropout_prob_list[3])
        self.en_4 = EncodeAndPredictLayer(512, 1024, num_class, True, concat_pred_list[4], share_list[4], dropout_prob_list[4])

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

        self.pred_de_3 = nn.Conv2d(64, num_class, 1)

    def forward(self, x_weak):
        # x should be nx3x160x160

        x_weak = x_weak
        x_strong = x_weak

        x_weak, x_strong, pred_en_0 = self.en_0(x_weak, x_strong)
        skip_0 = x_strong

        x_weak, x_strong, pred_en_1 = self.en_1(x_weak, x_strong)
        skip_1 = x_strong

        x_weak, x_strong, pred_en_2 = self.en_2(x_weak, x_strong)
        skip_2 = x_strong

        x_weak, x_strong, pred_en_3 = self.en_3(x_weak, x_strong)
        skip_3 = x_strong

        x_weak, x_strong, pred_en_4 = self.en_4(x_weak, x_strong)

        x = x_strong
        x = self.up_0(x)
        x = torch.cat([skip_3, x], 1)
        x = self.de_0(x)

        x = self.up_1(x)
        x = torch.cat([skip_2, x], 1)
        x = self.de_1(x)

        x = self.up_2(x)
        x = torch.cat([skip_1, x], 1)
        x = self.de_2(x)

        x = self.up_3(x)
        x = torch.cat([skip_0, x], 1)
        x = self.de_3(x)

        pred_de_3 = self.pred_de_3(x)

        return pred_en_0, pred_en_1, pred_en_2, pred_en_3, pred_en_4, pred_de_3
