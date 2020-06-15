""" Defines the neural net class DriftResNet. """

import torch
import torch.nn as nn

class DriftResNet(nn.Module):
    """
    Initialize to have a simple OU restoring drift
    """
    def __init__(self, hparams):
        super(DriftResNet, self).__init__()
        D = hparams.D
        H = hparams.H

        lin1 = nn.Linear(D, H)
        lin2 = nn.Linear(H, H)
        lin3 = nn.Linear(H, D)

        lin3.weight.data = torch.zeros(D, H)
        lin3.bias.data = torch.zeros(D)

        self.layers = nn.Sequential(lin1, nn.Hardtanh(), lin2, nn.Hardtanh(), lin3)

        self.res = nn.Linear(D, D, bias=False)
        self.res.weight.data = - torch.eye(D)  # Drift is restoring

    def forward(self, x):
        residual = x
        out = self.layers(x)
        out += self.res(residual)
        return out
