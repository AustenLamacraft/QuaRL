""" Defines the neural net class PairDriftHelium """

import torch
import torch.nn as nn

from continuum.nets.pairdrift import MakeFeatures, CombineFeatures


class PairDriftHelium(nn.Module):
    """ Neural net that uses the Equivariant PairDrift architecture.
    Uses a skip connection that takes advantage of Kato's cusp conditions.
    """

    def __init__(self, hparams):
        super(PairDriftHelium, self).__init__()

        self.make_features = MakeFeatures(hparams.D, hparams.H)
        self.combine1 = CombineFeatures(hparams.H, hparams.H)
        self.combine2 = CombineFeatures(hparams.H, hparams.D, zero_init=True)
        self.pair_lin = nn.Linear(hparams.H, hparams.H)

        def nucleus_cusp(x):
            """ Hydrogenic attraction to nucleus. """
            norm_x = torch.norm(x, dim=-1, keepdim=True)
            Z = 2
            return -Z * x / norm_x

        def electron_cusp(pairs):
            """ 'Hydrogenic' repulsion between electrons without decay """
            dx = pairs[..., 0:1, 1, :]  # i.e. x2 - x1
            norm_dx = torch.norm(dx, dim=-1, keepdim=True)
            v = -0.5 * dx / norm_dx
            return torch.cat((v, -v), dim=-2)

        self.activation = nn.Hardtanh()
        
        self.nucleus_cusp = nucleus_cusp
        self.electron_cusp = electron_cusp

    def forward(self, x):
        s, p, pairs = self.make_features(x)

        s = self.activation(s)
        p = self.activation(p)

        s = self.combine1(s, p)
        p = self.pair_lin(p)

        s = self.activation(s)
        p = self.activation(p)

        out = self.combine2(s, p)
        out += self.nucleus_cusp(x)
        out += self.electron_cusp(pairs)

        return out
    