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


'''
IDR: Information Decomposition and Reconstruction
'''


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
    def __init__(self, name1='resnet18', dim=128, K=1200, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 1200)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

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
        self.FM_pan = FM(in_channel=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.FM_ms = FM(in_channel=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.FM_high = FM(in_channel=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.FM_low = FM(in_channel=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.FM_fusion = FM(in_channel=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        # initialize encoder of key (PAN) by using encoder of query (Fusion) and not update by gradient
        for param_q, param_k in zip(
            self.EncoderF.parameters(), self.EncoderP.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # initialize encoder of key (MS) by using encoder of query (Fusion) and not update by gradient
        for param_q, param_k in zip(
            self.EncoderF.parameters(), self.EncoderM.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue_pan", torch.randn(dim, K))
        self.register_buffer("queue_ms", torch.randn(dim, K))
        self.queue_pan = nn.functional.normalize(self.queue_pan, dim=0)
        self.queue_ms = nn.functional.normalize(self.queue_ms, dim=0)

        # create the queue ptr
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # Momentum update of the key (PAN) encoder by using the query (Fusion) encoder
        for param_q, param_k in zip(
            self.EncoderF.parameters(), self.EncoderP.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        # Momentum update of the key (MS) encoder by using the query (Fusion) encoder
        for param_q, param_k in zip(
            self.EncoderF.parameters(), self.EncoderM.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        """
        :param keys1: keys is pan
        :param keys2: keys is ms
        :return:
        """
        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        # queue: [feature_dim, queue_length]; key: [batch_size, feature_dim]
        self.queue_ms[:, ptr: ptr + batch_size] = keys2.t()
        self.queue_pan[:, ptr: ptr + batch_size] = keys1.t()

        # move pointer after dequeuing and enqueuing
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr


    def forward(self, j_pan_l, j_ms_l, j_pan_h, j_ms_h):

        # pass the fusion module (FM) and get the fusion value
        fusion_pan = self.FM_pan(j_pan_h, j_pan_l)  # torch.Size([b, 64, 16, 16])
        fusion_ms = self.FM_ms(j_ms_h, j_ms_l)  # torch.Size([b, 64, 16, 16])
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
        # 第二个 encoder 最终需要将特征进行 flatten
        k_fusion = self.EncoderF(fusion_fusion, 2, 'up')  # queries: NxC
        k_fusion = nn.functional.normalize(k_fusion, dim=1)      # torch.Size([b, 128])

        # compute key features
        with torch.no_grad():                    # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_pan = self.EncoderP(fusion_pan, 2, 'up')
            k_ms = self.EncoderM(fusion_ms, 2, 'up')
            k_pan = nn.functional.normalize(k_pan, dim=1)        # torch.Size([b, 128])
            k_ms = nn.functional.normalize(k_ms, dim=1)          # torch.Size([b, 128])


        # compute logits
        # positive logits: Nx1
        # print(k_fusion.size())
        # print(k_pan.size())
        l_pos_fp = torch.einsum("nc,nc->n", [k_fusion, k_pan]).unsqueeze(-1)  # k_fusion as query, k_pan as key
        l_pos_fm = torch.einsum("nc,nc->n", [k_fusion, k_ms]).unsqueeze(-1)   # k_fusion as query, k_ms as key
        # negative logits: NxK
        l_neg_fp = torch.einsum("nc,ck->nk", [k_fusion, self.queue_pan.clone().detach()])  # k_fusion as query, k_pan as key
        l_neg_fm = torch.einsum("nc,ck->nk", [k_fusion, self.queue_ms.clone().detach()])   # k_fusion as query, k_ms as key

        # total logits: Nx(1+K)
        logits_fp = torch.cat([l_pos_fp, l_neg_fp], dim=1)
        logits_fm = torch.cat([l_pos_fm, l_neg_fm], dim=1)

        # apply temperature
        logits_fp /= self.T
        logits_fm /= self.T

        # labels: positive key indicators
        labels_fp = torch.zeros(logits_fp.shape[0], dtype=torch.long).cuda()
        labels_fm = torch.zeros(logits_fm.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_pan, k_ms)

        return logits_fp, logits_fm, labels_fp, labels_fm