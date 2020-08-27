import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import numpy as np

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # target = target.unsqueeze(1)
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()




def metric(probability, truth, prob_threshold, min_pix=0):
    '''Calculates dice per image'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth) 
    with torch.no_grad():

        
        t = truth.contiguous().view(batch_size, -1).float()  # torch.Size([4, 1048576])

        p = probability.contiguous().view(batch_size, -1)

        dice = torch.FloatTensor([0])
        for i in range(batch_size):
            p[i] = (p[i] > prob_threshold).float()
            if torch.equal(p[i], t[i]):
                dice = dice + 1
            else:  
                EPS = 1e-6
                intersection = torch.sum(p[i] * t[i])
                union = torch.sum(p[i]) + torch.sum(t[i]) + EPS
                dice = dice + (2*(intersection + EPS) / union ).mean()
    return dice/batch_size


