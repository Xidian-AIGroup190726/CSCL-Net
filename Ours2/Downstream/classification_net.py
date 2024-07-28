# import sys ,os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import numpy as np

from Pretrained.Models.IDRS import IDM
from Pretrained.Models.allResNets import resnet18
from Pretrained.Models.allResNets import resnet50


class CBR(nn.Module):
    def __init__(self, in_channel=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False):
        # print(in_channel)
        super(CBR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.avgpool = nn.AvgPool2d(3, stride=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.avgpool(x)
        return x


class FM(nn.Module):
    def __init__(self, in_channel=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False):
        super(FM, self).__init__()
        # the number of in_channel of CBR in FM = the number of input_1's channel + the number of input2's channel
        self.CBR = CBR(in_channel=in_channel,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       bias=bias)
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.CBR(x)

        return x


class IDR(nn.Module):
    def __init__(self, name1='resnet18', name='gaussian', D0=50, radius_ratio=0.5, length=50):
        super(IDR, self).__init__()

        # define encoder1 and encoder2
        if name1 == 'resnet18':
            self.EncoderPH = resnet18(version='V1', in_channel=64, width=1)
            self.EncoderPL = resnet18(version='V1', in_channel=64, width=1)
            self.EncoderMH = resnet18(version='V1', in_channel=64, width=1)
            self.EncoderML = resnet18(version='V1', in_channel=64, width=1)
        elif name1 == 'resnet50':
            self.EncoderPH = resnet50(version='V1', in_channel=1, width=1)
            self.EncoderPL = resnet50(version='V1', in_channel=1, width=1)
            self.EncoderMH = resnet50(version='V1', in_channel=4, width=1)
            self.EncoderML = resnet50(version='V1', in_channel=4, width=1)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name1))

        # define IDRs module
        self.IDMs = IDM(name=name, D0=D0, radius_ratio=radius_ratio, length=length)
        self.CBRph1 = CBR(in_channel=1, out_channels=64,kernel_size=3,stride=2,padding=1,bias=False)
        self.CBRph2 = CBR(in_channel=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.CBRph3 = CBR(in_channel=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.CBRpl1 = CBR(in_channel=1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.CBRpl2 = CBR(in_channel=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.CBRpl3 = CBR(in_channel=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.CBRmh = CBR(in_channel=4, out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.CBRml = CBR(in_channel=4, out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)


    def forward(self, pan, ms):
        # pass the IDRs module
        # tensor(cuda) to numpy
        pan_copy = pan.cpu().numpy()
        ms_copy = ms.cpu().numpy()
        h_pan_l = np.zeros_like(pan_copy)
        h_pan_h = np.zeros_like(pan_copy)
        h_ms_l = np.zeros_like(ms_copy)
        h_ms_h = np.zeros_like(ms_copy)
        # print(1)
        for i in range(pan.size(0)):
            # print(pan.size())
            # print('pan(copy) size is {}'.format(pan_copy[i,:,:,:].shape))  # (1, 64, 64)
            h_pan_l[i, :, :, :], h_pan_h[i, :, :, :] = self.IDMs.select(pan_copy[i, :, :, :], 1)
        for i in range(ms.size(0)):
            # print('ms(copy) size is {}'.format(ms_copy[i,:,:,:].shape))  # (4, 64, 64)
            h_ms_l[i, :, :, :], h_ms_h[i, :, :, :] = self.IDMs.select(ms_copy[i, :, :, :], 4)
        # numpy to tensor
        h_pan_l = torch.from_numpy(h_pan_l)
        h_pan_h = torch.from_numpy(h_pan_h)
        h_ms_l = torch.from_numpy(h_ms_l)
        h_ms_h = torch.from_numpy(h_ms_h)
        # set tensor.float()
        h_pan_l = h_pan_l.float()
        h_pan_h = h_pan_h.float()
        h_ms_l = h_ms_l.float()
        h_ms_h = h_ms_h.float()
        # set tensor.cuda()
        if torch.cuda.is_available():
            h_pan_l = h_pan_l.cuda()
            h_pan_h = h_pan_h.cuda()
            h_ms_l = h_ms_l.cuda()
            h_ms_h = h_ms_h.cuda()

        h_pan_h = self.CBRph1(h_pan_h)
        h_pan_h = self.CBRph2(h_pan_h)
        h_pan_h = self.CBRph3(h_pan_h)  # torch.Size([b, 64, 16, 16])
        h_pan_l = self.CBRpl1(h_pan_l)
        h_pan_l = self.CBRpl2(h_pan_l)
        h_pan_l = self.CBRpl3(h_pan_l)  # torch.Size([b, 64, 16, 16])

        h_ms_h = self.CBRmh(h_ms_h)     # torch.Size([b, 64, 16, 16])
        h_ms_l = self.CBRml(h_ms_l)     # torch.Size([b, 64, 16, 16])

        # pass the first encoder
        j_pan_h = self.EncoderPH(h_pan_h, 2)  # torch.Size([b, 128, 16, 16])
        j_pan_l = self.EncoderPL(h_pan_l, 2)  # torch.Size([b, 128, 16, 16])
        j_ms_h = self.EncoderMH(h_ms_h, 2)    # torch.Size([b, 128, 16, 16])
        j_ms_l = self.EncoderML(h_ms_l, 2)    # torch.Size([b, 128, 16, 16])

        return j_pan_l, j_ms_l, j_pan_h, j_ms_h


class MoCo(nn.Module):
    def __init__(self, name1='resnet18'):
        super(MoCo, self).__init__()

        # define encoder1 and encoder2
        if name1 == 'resnet18':
            self.EncoderP = resnet18(version='V2', in_channel=64, width=1)
            self.EncoderM = resnet18(version='V2', in_channel=64, width=1)
            self.EncoderF = resnet18(version='V2', in_channel=64, width=1)
        elif name1 == 'resnet50':
            self.EncoderP = resnet50(version='V2', in_channel=4, width=1)
            self.EncoderM = resnet50(version='V2', in_channel=4, width=1)
            self.EncoderF = resnet50(version='V2', in_channel=4, width=1)
        else:
            raise NotImplementedError('model {} is not implemented'.format(name1))

        # define fusion module
        self.FM_high = FM(in_channel=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.FM_low = FM(in_channel=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.FM_fusion = FM(in_channel=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, j_pan_l, j_ms_l, j_pan_h, j_ms_h):

        # pass the fusion module (FM) and get the fusion value
        # high: special info
        fusion_high = self.FM_high(j_pan_h, j_ms_h)  # torch.Size([b, 64, 16, 16])

        # j_pan_l1 = 0.5 * j_pan_l
        # j_ms_l1 = 0.5 * j_ms_l

        eps = torch.finfo(torch.float32).eps
        a = j_pan_l / (j_pan_l + j_ms_l + eps)
        b = 1 - a
        # print('a is {}'.format(a))
        # print('b is {}'.format(b))
        # print('j_pan_l is {}'.format(j_pan_l))
        # print('j_ms_l is {}'.format(j_ms_l))
        j_pan_l1 = a * j_pan_l
        # print('j_pan_ll is {}'.format(j_pan_l1))
        j_ms_l1 = b * j_ms_l
        # print('j_ms_l1 is {}'.format(j_ms_l1))

        # low: mutual info
        fusion_low = self.FM_low(j_pan_l1, j_ms_l1)  # torch.Size([b, 64, 16, 16])
        fusion_fusion = self.FM_fusion(fusion_high, fusion_low)  # torch.Size([b, 64, 16, 16])

        # pass the second encoder
        k_fusion = self.EncoderF(fusion_fusion, 2, 'down')  # queries: NxC
        # k_fusion = nn.functional.normalize(k_fusion, dim=1)      # torch.Size([b, 128])

        return k_fusion


class FCN(nn.Module):
    def __init__(self, dim):
        super(FCN, self).__init__()
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(4608, dim)
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, dim=1)  # torch.Size([b, 128])

        return x