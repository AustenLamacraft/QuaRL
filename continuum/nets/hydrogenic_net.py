""" Defines neural net class HydrogenicNet """

import torch
import torch.nn as nn


class HydrogenicNet(nn.Module):
    """
    Simple hydrogenic restoring drift
    """
    def __init__(self, hparams):
        super(HydrogenicNet, self).__init__()
        self.Z = nn.Parameter(torch.tensor(hparams.Z))

    def forward(self, x):
        return -self.Z * x / torch.norm(x, dim=-1, keepdim=True)
