from torch import nn
import torch
import torchvision
from torch.nn import functional as F
import pretrainedmodels

from . import blocks


class UNetResNextHyperSE(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNext: https://arxiv.org/abs/1611.05431

    Args:
    encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
    num_classes (int): Number of output classes.
    num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
    dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
    is_deconv (bool, optional):
        False: bilinear interpolation is used in decoder.
        True: deconvolution is used in decoder.
        Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 50:
            self.encoder = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        conv1.weight = self.encoder.layer0.conv1.weight

        self.input_adjust = blocks.EncoderBlock(
            nn.Sequential(conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1, self.pool),
            num_filters*2
        )

        self.conv1 = blocks.EncoderBlock(self.encoder.layer1, bottom_channel_nr//8)
        self.conv2 = blocks.EncoderBlock(self.encoder.layer2, bottom_channel_nr//4)
        self.conv3 = blocks.EncoderBlock(self.encoder.layer3, bottom_channel_nr//2)
        self.conv4 = blocks.EncoderBlock(self.encoder.layer4, bottom_channel_nr)

        self.dec4 = blocks.DecoderBlockV4(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = blocks.DecoderBlockV4(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = blocks.DecoderBlockV4(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = blocks.DecoderBlockV4(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.final = nn.Conv2d(2 * 2, num_classes, kernel_size=1)

        # deep supervision
        self.dsv4 = blocks.UnetDsv3(in_size=num_filters * 8, out_size=num_classes, scale_factor=8 * 2)
        self.dsv3 = blocks.UnetDsv3(in_size=num_filters * 8, out_size=num_classes, scale_factor=4 * 2)
        self.dsv2 = blocks.UnetDsv3(in_size=num_filters * 2, out_size=num_classes, scale_factor=2 * 2)
        self.dsv1 = nn.Conv2d(in_channels=num_filters * 2 * 2, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        center = self.conv4(conv3)

        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        # Deep Supervision
        dsv4 = self.dsv4(dec4)
        dsv3 = self.dsv3(dec3)
        dsv2 = self.dsv2(dec2)
        dsv1 = self.dsv1(dec1)
        dsv0 = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)

        return self.final(dsv0)


class UNetResNext(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNext: https://arxiv.org/abs/1611.05431

    Args:
    encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
    num_classes (int): Number of output classes.
    num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
    dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
    is_deconv (bool, optional):
        False: bilinear interpolation is used in decoder.
        True: deconvolution is used in decoder.
        Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 50:
            self.encoder = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.input_adjust = nn.Sequential(self.encoder.layer0.conv1,
                                          self.encoder.layer0.bn1,
                                          self.encoder.layer0.relu1)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = blocks.DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = blocks.DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = blocks.DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = blocks.DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)

        return self.final(dec1)


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385

    Args:
    encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
    num_classes (int): Number of output classes.
    num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
    dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
    pretrained (bool, optional):
        False - no pre-trained weights are being used.
        True  - ResNet encoder is pre-trained on ImageNet.
        Defaults to False.
    is_deconv (bool, optional):
        False: bilinear interpolation is used in decoder.
        True: deconvolution is used in decoder.
        Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):

        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = blocks.DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = blocks.DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = blocks.DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = blocks.DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)

        return self.final(dec1)


def get_model(model_type, **model_pars):
    """
    Create a new model

    :param model_type: str, type of model, one of ['UNet11', 'UNet16', 'LinkNet34', 'UNet']
    :param model_pars: dict, initialization parameters for the model
    :return: instance of nn.Module, a new Pytorch model
    """

    if model_type == 'UNetResNet34':
        return UNetResNet(encoder_depth=34, num_classes=1, pretrained=True,  num_filters=32, is_deconv=True, dropout_2d=0.2)

    if model_type == 'UNetResNet101':
        return UNetResNet(encoder_depth=101, num_classes=1, pretrained=True, num_filters=32, is_deconv=True, dropout_2d=0.2)

    if model_type == 'UNetResNet152':
        return UNetResNet(encoder_depth=152, num_classes=1, pretrained=True, num_filters=32, is_deconv=True, dropout_2d=0.2)

    if model_type == 'UNetResNext50':
        return UNetResNext(encoder_depth=50, num_classes=1, num_filters=16, is_deconv=True, dropout_2d=0)

    if model_type == 'UNetResNext101_32x4d':
        return UNetResNext(encoder_depth=101, num_classes=1, num_filters=32, is_deconv=True, dropout_2d=0.2)

    if model_type == 'UNetResNextHyperSE50':
        return UNetResNextHyperSE(encoder_depth=50, num_classes=1, num_filters=32, is_deconv=True, dropout_2d=0.2)

    if model_type == 'UNetResNextHyperSE101':
        return UNetResNextHyperSE(encoder_depth=101, num_classes=1, num_filters=32, is_deconv=True, dropout_2d=0.2)

    raise ValueError('Unknown model type: {}'.format(model_type))
