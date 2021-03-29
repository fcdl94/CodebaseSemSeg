import torch.nn as nn
import torch.nn.functional as F
import torch


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class HardNegativeMining(nn.Module):
    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, targets):
        # inputs should be B, H, W
        B = loss.shape[0]
        loss = loss.reshape(B, -1)
        P = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc*P))
        loss = tk[0].mean()
        return loss


