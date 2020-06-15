""" Neural net that uses the Equivariant DeepSets architecture. """

import torch
import torch.nn as nn


class PermEqMean(nn.Module):
    """ Returns equivariant layer used by EquivarDrift. """
    def __init__(self, in_dim, out_dim):
        super(PermEqMean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x_mean = x.mean(-2, keepdim=True)
        # x_mean = x_mean.expand(-1, x.shape[1], -1) # Not necessary due to broadcasting
        return self.Gamma(x) + self.Lambda(x_mean)


class EquivarDrift(nn.Module):
    """ Returns neural net that uses the Equivariant DeepSets architecture. """

    def __init__(self, d, num_channels):
        super(EquivarDrift, self).__init__()
        self.num_channels = num_channels
        self.d = d

        self.phi = nn.Sequential(
            PermEqMean(self.d, self.num_channels),
            nn.Hardtanh(inplace=True),
            PermEqMean(self.num_channels, self.num_channels),
            nn.Hardtanh(inplace=True),
            PermEqMean(self.num_channels, self.d),
        )

        self.phi[4].Gamma.weight.data = torch.zeros(d, num_channels)
        self.phi[4].Gamma.bias.data = torch.zeros(d)

        self.phi[4].Lambda.weight.data = torch.zeros(d, num_channels)

        self.res = nn.Linear(d, d, bias=False)
        self.res.weight.data = -torch.eye(d)  # Drift is restoring

    def forward(self, x):
        residual = x
        out = self.phi(x)
        out += self.res(residual)
        return out
 