""" Module for solving sdes.

Defines classes that perform one discrete time step of the sde per call.

Solvers:
  Euler: the Euler-Maruyama method.
  ExtrapolatedEuler: the extrapolated Euler-Maruyama method.
  Heun: Kloeden&Platen's scheme taylored for additive noise.
  SRA3: Stochastic Runge-Kutta method by Rossler.
  SOSRA: Stability Optimized Stochastic Runge-Kutta by Rackauckas and Nie.
"""

import math
import torch.nn as nn
import torch


class Euler(nn.Module):
    """
    Solving an SDE using the Euler-Maruyama method.

    Takes drift function v(X) and updates according to
    X_{t+1} = X_{t} + Delta W_t + v(X_t) Delta t
    """
    def __init__(self, drift_fn, _):
        super(Euler, self).__init__()
        self.drift_fn = drift_fn

    def forward(self, state, dt):
        dW = torch.randn_like(state) * math.sqrt(dt)
        drift = self.drift_fn(state)
        dstate = dW + drift * dt

        state = state + dstate
        return state, drift, dW


class ExtrapolatedEuler(nn.Module):
    """
    Solving an SDE using the weak order 2.0 extrapolated Euler-Maruyama method.
    Uses two EM steps: one with half the Delta t of the other and
    extrapolates the results to calculate log_rn.
    Uses common random numbers to reduce memory and variance.
    """
    def __init__(self, drift_fn, _):
        super(ExtrapolatedEuler, self).__init__()
        self.drift_fn = drift_fn

    def forward(self, state, dt):
        """
        Variables with trailing underscores relate to the rougher step of dt.
        Variables without trailing underscore relate to the 2 finer steps dt/2.
        The variable log_rn_f refers to the 'final' estimation of log_rn
        """
        state_1, state_1_ = torch.chunk(state, 2)
        drift_1, drift_1_ = torch.chunk(self.drift_fn(state), 2)

        # Two Euler steps with step size dt / 2
        dW_1 = torch.randn_like(state_1) * math.sqrt(dt / 2)
        state_2 = state_1 + dW_1 + drift_1 * (dt / 2)

        dW_2 = torch.randn_like(state_2) * math.sqrt(dt / 2)
        drift_2 = self.drift_fn(state_2)
        state_3 = state_2 + dW_2 + drift_2 * (dt / 2)

        # Euler step with step size dt
        dW_1_ = dW_1 + dW_2
        state_3_ = state_1_ + dW_1_ + drift_1_ * dt

        # Combine
        state = torch.cat((state_3, state_3_))
        drift = torch.cat((drift_1, drift_1_))

        return state, drift, dW

class Heun(nn.Module):
    """
    Solving an SDE using a stochastic "Heun's" method.
    This is the "multi-dimensional explicit order 2.0 weak scheme"
    for additive noise, see paragraph 15.1 (Explicit Order 2.0 Weak Schemes)
    of Kloeden and Platen's "Numerical Solution of Stochastic Differential
    Equations".
    """
    def __init__(self, drift_fn, _):
        super(Heun, self).__init__()
        self.drift_fn = drift_fn

    def forward(self, state, dt):
        dW = torch.randn_like(state) * math.sqrt(dt)

        drift_1 = self.drift_fn(state)
        state_2 = state + dW + drift_1 * dt
        drift_2 = self.drift_fn(state_2)

        drift_tot = 0.5 * (drift_1 + drift_2)
        dstate = dW + drift_tot * dt

        state = state + dstate
        return state, drift_1, dW


class SRA3(nn.Module):
    """
    Solving an SDE using a (1.5, 3.0) order method: SRA3 [1].

      [1]: Rößler, Andreas.
      "Runge–Kutta methods for the strong approximation
      of solutions of stochastic differential equations."
      SIAM Journal on Numerical Analysis 48.3 (2010): 922-952.
      https://doi.org/10.1137/09076636X
    """
    def __init__(self, drift_fn, hparams):
        super(SRA3, self).__init__()
        self.drift_fn = drift_fn
        self.strong = hparams.strong

    def forward(self, state, dt):
        sqrtdt = math.sqrt(dt)
        dW = torch.randn_like(state) * sqrtdt
        if self.strong:
            dZ = 0.5*(dW + torch.randn_like(state) * sqrtdt / math.sqrt(3.0))
        else:
            dZ = 0.5 * dW
            s = 2 * torch.randint_like(state, 2) - 1
            dZ += 0.5 * s * sqrtdt / math.sqrt(3.0)

        drift_1 = self.drift_fn(state)
        state_2 = state + drift_1 * dt
        drift_2 = self.drift_fn(state_2)
        state_3 = state + 1.50 * dZ + (0.25 * drift_1 + 0.25 * drift_2) * dt
        drift_3 = self.drift_fn(state_3)

        drift_tot = (drift_1/6.0 + drift_2/6.0 + drift_3*2.0/3.0)
        dstate = dW + drift_tot * dt

        state = state + dstate
        return state, drift_1, dW


class SOSRA(nn.Module):
    """
    Solving an SDE using a stability optimized SRA method of order (1.5, 2.0) called SOSRA [1].

      [1] Rackauckas, Christopher and Qing Nie.
      “Stability-Optimized High Order Methods and Stiffness Detection
      for Pathwise Stiff Stochastic Differential Equations.”
      (2018).
      https://pdfs.semanticscholar.org/d676/86b04ee58c1656b611d2d1ce6b12fc452a9e.pdf
    """
    def __init__(self, drift_fn, hparams):
        super(SOSRA, self).__init__()
        self.drift_fn = drift_fn
        self.strong = hparams.strong

    def forward(self, state, dt):
        sqrtdt = math.sqrt(dt)
        dW = torch.randn_like(state) * sqrtdt

        if self.strong:
            dZ = 0.5*(dW + torch.randn_like(state) * sqrtdt / math.sqrt(3.0))
        else:
            dZ = 0.5 * dW
            s = 2 * torch.randint_like(state, 2) - 1
            dZ += 0.5 * s * sqrtdt / math.sqrt(3.0)

        drift_1 = self.drift_fn(state)
        state_2 = state + 1.3371632704399763*dZ + 0.6923962376159507*drift_1*dt
        drift_2 = self.drift_fn(state_2)
        state_3 = (state + 3.30564519858*dZ
                   + (-3.1609142252828395*drift_1 + 4.1609142252828395*drift_2)*dt)
        drift_3 = self.drift_fn(state_3)

        drift_tot = (0.2889874966892885*drift_1 + 0.6859880440839937*drift_2
                     + 0.025024459226717772*drift_3)
        dstate = dW + drift_tot * dt

        state = state + dstate
        return state, drift_1, dW
