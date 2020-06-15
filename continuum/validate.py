""" Defines the class Validator."""

import math
import torch


class Validator:
    """
    Validates the energy of model with bias and variance below given tolerances.

    Args (of __init__):
      model (continuum.Model): the drift to evaluate

    Args (of __call__):
      atol_bias (Float): absolute tolerance for the bias.
          Validation will continue until the bias estimate is smaller than 1.1*atol_bias.
      atol_var (Float): absolute tolerance for the variance.
          Validation will continue until the variance estimate is smaller than 1.1*atol_var.
          If not given, atol_var will be set to (atol_bias)**2,
          so that error due to variance is equal to atol_bias.
      state (Tensor): an initial state.
          A good initial state is the latest state computed during training.
          If not given, evaluation will start with a Gaussian state.

    Returns:
      avg_energy (Float): the evaluated energy
      bias (Float): estimated bias of the evaluated energy
      var (Float): estimated variance of the evaluated energy
    """

    def __init__(self, model, atol_bias, atol_var=None):
        self.model = model
        self.h = model.hparams
        self.dt = self.h.dt
        self.num_steps = self.h.num_steps

        # order of convergence assumed for bias estimate
        self.order = 2.0 if self.h.method in {'SOSRA', 'SRA3'} else 1.0

        self.atol_bias = torch.tensor(atol_bias)
        self.atol_var = (torch.tensor(atol_var)
                         if atol_var is not None
                         else self.atol_bias**2)

        # use dt and 0.5*dt to compute bias of dt
        self.dt_multiplier = 0.5

        self.state = None
        self.stages = 1  # number of stages to compute to avoid CUDA out of memory

        self.energy_fn = globals()[self.h.validate_fn](model.potential_fn)

    def __call__(self, state, talks=False, max_stages=2):
        """
        Validates the energy of model with state as initial state.

        Kwargs:
          - talks (Bool): controls whether progress is printed
          - max_stages (int): validation is done in stages to avoid going out of memory.
                The number of stages doubles as dt halves. Setting max_stages effectively
                limits the time Validator can take: recommended values are max_stages=2
                during training and max_stages=8 for validation after training.
         """

        h = self.h
        self.dt = h.dt
        self.num_steps = h.num_steps
        self.stages = 1

        state = self._preprocess(state)

        avg_energy = torch.tensor(float('inf'))
        bias = torch.tensor(float('inf'))
        var = torch.tensor(float('inf'))

        while ((torch.abs(bias) > self.atol_bias * 1.2
               or var > self.atol_var * 1.2)
               and self.stages <= max_stages):

            if str(state.device).startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated(device=state.device)

            old_energy = avg_energy

            energy = 0

            for i in range(self.stages):
                if talks:
                    print('Stage', i+1, 'of', self.stages)
                energy += self._calculate_energy(state)

            avg_energy = energy.mean() / self.stages

            bias = self._calculate_bias(old_energy, avg_energy)
            var = self._calculate_var(energy)

            if talks:
                print('energy', float(avg_energy))
                print('bias of previous energy', float(bias))
                print('std', float(torch.sqrt(var)), ', target', float(torch.sqrt(self.atol_var)))
                print('dt', self.dt)
                print('batch size', state.shape[0])
                print('num_steps', self.num_steps)
                print('\n')

            state = self._update_hparams(state, var)

        return avg_energy

    @staticmethod
    def _preprocess(state):
        """
        Ensures correct shape of state.
        """
        state = state.squeeze(dim=0)
        if state.shape[0] % 32 != 0:
            state = state.repeat([32, 1, 1])

        return state

    def _calculate_energy(self, state):
        state_traj, _, _ = self.model.forward(state, self.num_steps, self.dt)
        state = state_traj[-1]  # make stationary

        state_traj, drift_traj, dW_traj = self.model.forward(state, self.num_steps, self.dt)
        return self.energy_fn(state_traj, drift_traj, dW_traj, self.dt)

    def _calculate_bias(self, old_energy, new_energy):
        """ Calculates bias of old_energy using new_energy """
        if self.dt_multiplier < 0.99 and not torch.isinf(old_energy):
            bias = (old_energy - new_energy) * (
                1 / (1 - self.dt_multiplier**(self.order)))
        else:
            bias = torch.tensor(float('inf'))

        return bias

    @staticmethod
    def _calculate_var(energy):
        """ Calculates variance of mean in robust way.
        Divides batch in 32 pieces and calculate variance between those 32 pieces.
        """
        energies = energy.chunk(32)
        means = torch.tensor([chunk.mean() for chunk in energies])
        return means.var() / 32

    def _update_hparams(self, state, var):
        """ Update dt and batch_size, keep T constant. """
        self.dt = self.dt * self.dt_multiplier
        self.num_steps = round(self.num_steps / self.dt_multiplier)
        
        if str(state.device).startswith('cuda'):
            # change batch size (also to be able to estimate biases)
            batch_size_multiplier = float(max(var / (self.atol_bias / 8)**2,
                                          var / self.atol_var, 1.0))

            # get used and available memory
            t = torch.cuda.get_device_properties(state.device).total_memory
            a = torch.cuda.max_memory_reserved(device=state.device)
            torch.cuda.reset_max_memory_allocated(device=state.device)

            # If wants to grow more than possible (factor 2 because num_steps *= 2)
            if batch_size_multiplier > t / a / 2:
                # multiply with maximum factor, keep on safe side
                batch_size_multiplier = t / a / 2 * 0.95

            if batch_size_multiplier < 1.0:  # if it needs to shrink
                cutoff = round(batch_size_multiplier * state.shape[0])
                state = state[0:cutoff]
            else:
                # compute maximum possible batch size
                max_batch_size = math.floor(batch_size_multiplier * state.shape[0])
                # repeat state so that batch_size > max_batch_size
                state = state.repeat([math.ceil(batch_size_multiplier), 1, 1])
                # cut off state so that batch_size = max_batch_size
                state = state[0:max_batch_size]

            self.stages *= 2

        return state


def holland_energy(potential_fn):
    """ Calculates the log-likelihood ell of a trajectory """

    def energy_fn(state_traj, drift_traj, dW_traj, dt):
        potential = potential_fn(state_traj[:-1])

        kinetic = + 0.5 * torch.sum(drift_traj[:-1]**2, dim=(-1, -2))
        kinetic += torch.sum(drift_traj[:-1] * dW_traj / dt, dim=(-1, -2))
        energy = (potential + kinetic).mean(dim=0)  # average over time
        return energy

    return energy_fn


def stratonovich_energy(potential_fn):
    """ Calculates the log-likelihood tilde{ell} of a trajectory """

    def energy_fn(state_traj, drift_traj, dt):
        potential = potential_fn(state_traj[:-1])
        dXs = state_traj[1:] - state_traj[:-1]

        kinetic = - 0.5 * torch.sum(drift_traj[:-1]**2, dim=(-1, -2))
        kinetic += 0.5 * torch.sum((drift_traj[:-1] - drift_traj[1:]) * dXs/dt, dim=(-1, -2))

        energy = (potential + kinetic).mean(dim=0)  # average over time
        return energy

    return energy_fn
