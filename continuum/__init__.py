""" Package for continuum models.

Usage:
  Create custom hparams from continuum.default_hparams,
  and pass this into continuum.Model.

Submodules:
  evaluate: calculates evaluated losses for models.
  model: defines continuum.Model
  potentials: defines potential energy functions that can be used by SDESolve.
  sdesolvers: defines sde discrete time approximation schemes
  test: testing suite

Subpackage:
  nets: contains neural nets appropriate for different continuum models.
"""

import continuum.losses
import continuum.nets
import continuum.potentials
import continuum.sdesolvers
import continuum.validate

from .model import Model, DEFAULT_HPARAMS
