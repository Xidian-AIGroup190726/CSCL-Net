import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torch.nn import functional as F


# allResNets includes 'resnet18' and 'resnet50'


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):  # BasicBlock is the residual block of Resnet18 and Resnet34
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):  # inplanes, planes --> in_planes, out_planes
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # Bottleneck is the residual block of resnet50, resnet101 and resnet152
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        :param in_planes: the input channel's number of the first convolution operation in the bottleneck
        :param planes: the output channel's number of the first convolution operation in the bottleneck
        :param stride:
        :param downsample:
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetV1(nn.Module):
    def __init__(self, block, layers, in_channel=3, width=1):
        """
        :param block: residual block,
        :param layers: layers is a list, layers[i] is the number of residual block of the ith layer
        :param in_channel: input channels of the first convolutional layer of the residual network
        :param width:
        """
        # print(in_channel)
        # print(width)
        self.in_planes = 64
        super(ResNetV1, self).__init__()
        # self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        """
        self.layer1: the first residual layer of resnet,   planes = self.base
        self.layer2: the second residual layer of resnet,  planes = self.base * 2
        self.layer3: the third residual layer of resnet,   planes = self.base * 3
        self.layer4: the fourth residual layer of resnet,  planes = self.base * 4
        """
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=1)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=1)
        # self.avgpool = nn.AvgPool2d(3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block: the residual block
        :param planes: the output channel's number of the first convolution operation in the basic block or bottleneck
        :param blocks: the number of residual block of the ith layer
        :param stride:
        :return:
        """
        downsample = None

        """
        self.inplanes: the input channel's number of the basic block or the bottleneck
        planes: the output channel's number of the first convolution operation in the basic block or bottleneck
        planes * block.expansion: the output channel's number of the basic block or bottleneck
        """

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        for i in range(1, blocks):
            """
            In each residual network only the first residual block is involved in the downsampling operation 
            """
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, layer=2):
        if layer <= 0:
            return x
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if layer == 1:
            return x
        x = self.layer2(x)
        if layer == 2:
            return x
        x = self.layer3(x)
        if layer == 3:
            return x
        x = self.layer4(x)
        if layer == 4:
            return x
        # x = self.avgpool(x)
        # if layer == 5:
        #     return x

        return x


# ResnetV3(Encoder): Feature encoding module for contrastive learning
class ResNetV2(nn.Module):
    def __init__(self, block, layers, low_dim=128, in_channel=3, width=1):
        """
        :param block: residual block,
        :param layers: layers is a list, layers[i] is the number of residual block of the ith layer
        :param low_dim: the dimensionality of the output layer
        :param in_channel: input channels of the first convolutional layer of the residual network
        :param width:
        """
        self.in_planes = 64
        super(ResNetV2, self).__init__()
        # self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        """
        self.layer1: the first residual layer of resnet,   planes = self.base
        self.layer2: the second residual layer of resnet,  planes = self.base * 2
        self.layer3: the third residual layer of resnet,   planes = self.base * 3
        self.layer4: the fourth residual layer of resnet,  planes = self.base * 4
        """
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(4608, low_dim)
        # self.l2norm = Normalize(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block: the residual block
        :param planes: the output channel's number of the first convolution operation in the basic block or bottleneck
        :param blocks: the number of residual block of the ith layer
        :param stride:
        :return:
        """
        downsample = None

        """
        self.inplanes: the input channel's number of the basic block or the bottleneck
        planes: the output channel's number of the first convolution operation in the basic block or bottleneck
        planes * block.expansion: the output channel's number of the basic block or bottleneck
        """

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        for i in range(1, blocks):
            """
            In each residual network only the first residual block is involved in the downsampling operation 
            """
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, layer=2, task='up'):
        if layer <= 0:
            return x
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print('After passing the first layer, the size is {}'.format(x.size()))
        if layer == 1:
            return x

        x = self.layer2(x)
        # print('{}'.format(x.size()))
        if layer == 2 and task == 'down':
            return x
        elif layer == 2 and task == 'up':
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            # x = self.l2norm(x)
            return x

        x = self.layer3(x)
        # print('{}'.format(x.size()))
        if layer == 3:
            return x

        x = self.layer4(x)
        # print('{}'.format(x.size()))
        if layer == 4:
            return x

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.l2norm(x)
        # if layer == 5:
        #     return x
        return x


# def __init__(self, block, layers, low_dim=128, in_channel=3, width=1):
def resnet18(version='V1', pretrained=False, **kwargs):
    """
    :param version:
    :param pretrained: If True, returns a model pretrained on ImageNet
    :param kwargs: {'in_channel': value, 'width': value}
    :return:
    """
    # print(kwargs)
    if version == 'V1':
        model = ResNetV1(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif version == 'V2':
        model = ResNetV2(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(version='V1', pretrained=False, **kwargs):
    """
    :param version:
    :param pretrained: If True, returns a model pretrained on ImageNet
    :param kwargs: {'in_channel': value, 'width': value}
    :return:
    """
    # print(kwargs)
    if version == 'V1':
        model = ResNetV1(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif version == 'V2':
        model = ResNetV2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

# x = torch.ones(4, 64, 16, 16)
# EncoderPH = resnet18(version='V1', in_channel=64, width=1)
# x = EncoderPH(x, 2)
# print(x.size())  # torch.Size([4, 512, 16, 16])

# x = torch.ones(4, 64, 16, 16)
# EncoderP = resnet18(version='V2', in_channel=64, width=1)
# x = EncoderP(x)
# print(x.size())