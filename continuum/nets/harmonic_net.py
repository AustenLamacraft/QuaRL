""" Defines neural net class HarmonicNet """

import torch
import torch.nn as nn


class HarmonicNet(nn.Module):
    """
    Simple OU restoring drift
    """
    def __init__(self, hparams):
        super(HarmonicNet, self).__init__()
        self.Z = nn.Parameter(torch.tensor(hparams.Z))

    def forward(self, x):
        return -self.Z * x
