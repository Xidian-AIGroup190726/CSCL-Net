import torch

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]"""
    N, C, _, _ = img1.shape

    # reshape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)

    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    # print(cc)

    cc = torch.clamp(cc, -1., 1.)

    return cc.mean()

#
# x1 = torch.randn((4, 64, 8, 16))
# x2 = torch.randn((4, 64, 8, 16))
# res = cc(x1, x2)
# print(res)
# print(res.size())  # torch.Size([])