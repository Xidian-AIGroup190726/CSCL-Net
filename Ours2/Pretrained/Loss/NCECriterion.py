import torch
from torch import nn

"""
if self.softmax = True,  using criterion of NCESoftmaxLoss, corresponding to the end version_1 of out_l and out_ab
"""

"""
There is one ending version of out_1 and out_ab.
End version_1 is correspond to the situation that the self.softmax is equal to True.

ending version_1 as follows:
If self.use_softmax = True,
the end version_1 of out_l and out_ab are equal to the preliminary version dividing t,
and criterion_l and criterion_ab are the object of NCESoftmaxLoss.
"""

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-Loss loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss
