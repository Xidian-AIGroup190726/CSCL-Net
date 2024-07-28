import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):
    """
    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        """
        :param inputSize: '--feat_dim', type=int, default=128, help='dim of feat for inner product',
                          the feature dimensionality stored memory bank or queue
        :param outputSize: n_data, len(dataset)
        :param K: '--nce_k', type=int, default=16384, the number of negative sample
        :param T: '--nce_t', type=float, default=0.07, the temperature factor of NCELoss
        :param momentum: '--nce_m', type=float, default=0.5
        :param use_softmax: '--softmax', action='store_true', help='using softmax contrastive loss rather than Loss'
        """
        super(NCEAverage, self).__init__()
        self.nLem = outputSize  # n_data, which is the length of all train data sets, and is the length of memory bank
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)  # The function of AliasMethod is to take random negative samples
        self.multinomial.cuda()
        self.K = K  # the number of negative sample
        self.use_softmax = use_softmax

        # define the self.params and that are equal to torch.tensor([K, T, -1, -1, momentum])
        self.register_buffer('params', torch.tensor([K, T, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)

        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

        self.register_buffer('memory_v3', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    # in train_CMC: out_pan, out_ms, out_f_pan, out_f_ms = contrast(final_feat_pan, final_feat_ms, final_feat_f, index)
    def forward(self, v1, v2, v3, y, idx=None):
        """
        :param v1: pan modality
        :param v2: ms modality
        :param v3: f modality
        :param y: index of batch data
        :param idx:
        :return:
        """
        K = int(self.params[0].item())  # the number of negative sample
        T = self.params[1].item()       # the temperature factor of NCELoss
        momentum = self.params[2].item()

        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)  # the number of sample of memory bank
        inputSize = self.memory_v1.size(1)   # the feature dimensionality

        # score computation
        if idx is None:
            # sample idx of positive and negatives for each sample of batch
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)  # idx.size: (batchSize, K + 1)
            # define the idx of positive for each sample of batch
            idx.select(1, 0).copy_(y.data)

        """
        out_vx: (batchSize, K + 1, 1), vx as anchor, 
        vx as anchor, and get the dot product between the anchor point and positive and negative samples
        """

        # sample weights of v3(f) corresponding to v1(pan) and get out_v1
        weight_v3 = torch.index_select(self.memory_v3, 0, idx.view(-1)).detach()

        weight_v3 = weight_v3.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v3, v1.view(batchSize, inputSize, 1))  # out_v1.size: (batchSize, K + 1, 1)

        # sample weights of v3(f) corresponding to v2(ms) and get out_v2
        weight_v3 = torch.index_select(self.memory_v3, 0, idx.view(-1)).detach()

        weight_v3 = weight_v3.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v3, v2.view(batchSize, inputSize, 1))  # out_v1.size: (batchSize, K + 1, 1)

        # sample weights of v1(pan) corresponding to v3(f) and get out_v31(f)
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()

        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v31 = torch.bmm(weight_v1, v3.view(batchSize, inputSize, 1))  # out_v2.size: (batchSize, K + 1, 1)

        # sample weights of v2(ms) corresponding to v3(f) and get out_v32
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()

        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v32 = torch.bmm(weight_v2, v3.view(batchSize, inputSize, 1))  # out_v1.size: (batchSize, K + 1, 1)
        # provide the end version of out _v1 and out_v2 as follows:
        if self.use_softmax:
            out_v1 = torch.div(out_v1, T)
            out_v2 = torch.div(out_v2, T)
            out_v31 = torch.div(out_v31, T)
            out_v32 = torch.div(out_v32, T)
            out_v1 = out_v1.contiguous()
            out_v2 = out_v2.contiguous()
            out_v31 = out_v31.contiguous()
            out_v32 = out_v32.contiguous()

        # update memory bank
        with torch.no_grad():
            v1_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            v1_pos.mul_(momentum)
            v1_pos.add_(torch.mul(v1, 1 - momentum))
            v1_norm = v1_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = v1_pos.div(v1_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            v2_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            v2_pos.mul_(momentum)
            v2_pos.add_(torch.mul(v2, 1 - momentum))
            v2_norm = v2_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = v2_pos.div(v2_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

            v3_pos = torch.index_select(self.memory_v3, 0, y.view(-1))
            v3_pos.mul_(momentum)
            v3_pos.add_(torch.mul(v3, 1 - momentum))
            v3_norm = v3_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v3 = v3_pos.div(v3_norm)
            self.memory_v3.index_copy_(0, y, updated_v3)
        return out_v1, out_v2, out_v31, out_v32