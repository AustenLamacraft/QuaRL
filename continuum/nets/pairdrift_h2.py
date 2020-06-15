""" Defines the neural net class PairDriftH2 """

import torch
import torch.nn as nn

from continuum.nets.pairdrift import MakeFeatures, CombineFeatures


class PairDriftH2(nn.Module):
    """ Neural net that uses the Equivariant PairDrift architecture.
    Uses a skip connection that takes advantage of Kato's cusp conditions.
    """

    def __init__(self, hparams):
        super(PairDriftH2, self).__init__()

        self.make_features = MakeFeatures(hparams.D, hparams.H)
        self.combine1 = CombineFeatures(hparams.H, hparams.H)
        self.combine2 = CombineFeatures(hparams.H, hparams.D, zero_init=True)
        self.pair_lin = nn.Linear(hparams.H, hparams.H)

        def nucleus_cusp(x):
            """
            'Hydrogenic' repulsion between electron and protons without decay.
            """
            if not hasattr(self, 'p1'):                
                self.p1 = torch.tensor([0.0, 0.0, hparams.R / 2], device=x.device)
                self.p2 = -self.p1
            
            dx1, dx2 = x - self.p1, x - self.p2
            norm_dx1 = torch.norm(dx1, dim=-1, keepdim=True)
            norm_dx2 = torch.norm(dx2, dim=-1, keepdim=True)
            v1 = -dx1 / norm_dx1
            v2 = -dx2 / norm_dx2

            v = v1 + v2
            return v

        def electron_cusp(pairs):
            """
            'Hydrogenic' repulsion between electrons without decay
            """
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
