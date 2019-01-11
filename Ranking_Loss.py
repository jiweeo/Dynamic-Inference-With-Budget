import numpy as np
import torch
import torch.nn as nn

class Ranking_Loss(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.num_block = num_blocks
        self.margin = 1. / num_blocks

    def forward(self, pred, target):

        # compare function
        target_a = torch.unsqueeze(target,dim=1).repeat(1, self.num_block, 1)
        target_b = torch.unsqueeze(target,dim=2).repeat(1, 1, self.num_block)
        phi = (target_a > target_b).float()

        # difference
        pred_a = torch.unsqueeze(pred, dim=1).repeat(1, self.num_block, 1)
        pred_b = torch.unsqueeze(pred, dim=2).repeat(1, 1, self.num_block)
        diff = pred_b - pred_a + self.margin
        diff = torch.clamp(diff, min=0.0)

        # compute loss
        loss = diff * phi

        # pick mean
        return loss.sum()


if __name__ == '__main__':
    # for testing
    criterion = Ranking_Loss(54)
    pred = np.random.randint(54, size=(32, 54))
    target = np.random.randn(32, 54)
    loss = criterion(pred, target)
