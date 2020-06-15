"""Module for potentials

The shape of the data is assumed to be [batch, particle, dimension]
"""

import math
import torch


def sho_potential(x):
    """
    The simple harmonic oscillator
    """
    return 0.5*torch.sum(x**2, dim=(-1, -2))


def h_potential(x):
    """
    The Hydrogen atom
    """
    return -1 / torch.sqrt(torch.sum(x**2, dim=(-1, -2)))


def he_potential(x):
    """
    The Helium atom
    """
    x1, x2 = torch.chunk(x, 2, dim=-2)
    u1 = 1 / torch.sqrt(torch.sum(x1**2, dim=(-1, -2)))
    u2 = 1 / torch.sqrt(torch.sum(x2**2, dim=(-1, -2)))
    u12 = 1 / torch.sqrt(torch.sum((x1-x2)**2, dim=(-1, -2)))
    return -2 * u1 - 2 * u2 + u12



def h2_potential(x):
    """
    The H2 molecule
    """
    R = 1.401 # Equilibrium separation of the protons in Bohr radii

    p1 = torch.tensor([[[0.0, 0.0, R / 2]]], dtype=torch.float).to(x.device)
    p2 = torch.tensor([[[0.0, 0.0, -R / 2]]], dtype=torch.float).to(x.device)

    x1, x2 = torch.chunk(x, 2, dim=-2)
    u11 = 1 / torch.sqrt(torch.sum((x1-p1)**2, dim=(-1, -2)))
    u12 = 1 / torch.sqrt(torch.sum((x1-p2)**2, dim=(-1, -2)))
    u21 = 1 / torch.sqrt(torch.sum((x2-p1)**2, dim=(-1, -2)))
    u22 = 1 / torch.sqrt(torch.sum((x2-p2)**2, dim=(-1, -2)))

    uee = 1 / torch.sqrt(torch.sum((x1-x2)**2, dim=(-1, -2)))
    upp = 1 / R

    return - u11 - u12 - u21 - u22 + uee + upp


def h2_param(hparams):
    """
    The H2 molecule with variable proton separation
    """

    R = getattr(hparams, 'R', 1.401)

    def h2_pot(x):

        p1 = torch.tensor([[[0.0, 0.0, R / 2]]], dtype=torch.float).to(x.device)
        p2 = torch.tensor([[[0.0, 0.0, -R / 2]]], dtype=torch.float).to(x.device)

        x1, x2 = torch.chunk(x, 2, dim=-2)
        u11 = 1 / torch.sqrt(torch.sum((x1-p1)**2, dim=(-1, -2)))
        u12 = 1 / torch.sqrt(torch.sum((x1-p2)**2, dim=(-1, -2)))
        u21 = 1 / torch.sqrt(torch.sum((x2-p1)**2, dim=(-1, -2)))
        u22 = 1 / torch.sqrt(torch.sum((x2-p2)**2, dim=(-1, -2)))

        uee = 1 / torch.sqrt(torch.sum((x1-x2)**2, dim=(-1, -2)))
        upp = 1 / R

        return - u11 - u12 - u21 - u22 + uee + upp

    return h2_pot




def gaussint_param(hparams):
    """
    Parameterized bosons in a harmonic trap with Gaussian interaction
    """

    g, s, omega = hparams.g, hparams.s, hparams.omega
    number_of_particles = hparams.number_of_particles

    def gaussint(x):

        # Note that torch.pdist doesn't yet support batch mode so we use cdist
        # See https://github.com/pytorch/pytorch/issues/9406
        # (there is double counting to correct for)
        x_ = x.view((-1,) + x.shape[-2:])
        pd = torch.cdist(x_, x_, p=2)
        pd = pd.view(x.shape[:-1] + (number_of_particles,))

        gauss_int = g / (math.pi * s ** 2) * torch.exp(- pd**2 / s**2)
        interaction = (1 / 2) * torch.sum(gauss_int, dim=(-1, -2))
        interaction -= 0.5 * number_of_particles * gauss_int[..., 0, 0]
        potential = (1 / 2) * omega**2 * torch.sum(x**2, dim=(-1, -2))

        return potential + interaction

    return gaussint

