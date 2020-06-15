""" Defines neural net class PairDrift. """


import torch
import torch.nn as nn


class MakeFeatures(nn.Module):
    """ Returns features to be used by PairDrift. """
    def __init__(self, in_dim, out_dim):
        super(MakeFeatures, self).__init__()
        self.single = nn.Linear(in_dim, out_dim)
        self.pair = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        pairs = x[..., None, :, :] - x[..., :, None, :]
        return self.single(x), self.pair(pairs), pairs


class CombineFeatures(nn.Module):
    """ Returns layer to be used by PairDrift. """
    def __init__(self, in_dim, out_dim, zero_init=False):
        super(CombineFeatures, self).__init__()
        self.single = nn.Linear(in_dim, out_dim)
        self.pair = nn.Linear(in_dim, out_dim)

        if zero_init:
            self.single.weight.data = torch.zeros(out_dim, in_dim)
            self.single.bias.data = torch.zeros(out_dim)

            self.pair.weight.data = torch.zeros(out_dim, in_dim)
            self.pair.bias.data = torch.zeros(out_dim)

    def forward(self, s, p):
        return self.single(s) + self.pair(p).sum(dim=-3)


class PairDrift(nn.Module):
    """ Returns neural net that uses the Equivariant PairDrift architecture. """

    def __init__(self, hparams):
        super(PairDrift, self).__init__()
        d = hparams.D
        num_channels = hparams.H

        self.make_features = MakeFeatures(d, num_channels)
        self.combine1 = CombineFeatures(num_channels, num_channels)
        self.combine2 = CombineFeatures(num_channels, d, zero_init=True)
        self.pair_lin = nn.Linear(num_channels, num_channels)
        
        self.activation = nn.Hardtanh()

        self.skip = nn.Linear(d, d, bias=False)
        self.skip.weight.data = -torch.eye(d)  # Drift is restoring

    def forward(self, x):
        s, p, _ = self.make_features(x)

        s = self.activation(s)
        p = self.activation(p)

        s = self.combine1(s, p)
        p = self.pair_lin(p)

        s = self.activation(s)
        p = self.activation(p)

        out = self.combine2(s, p)
        out += self.skip(x)

        return out
    