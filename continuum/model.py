"""
Module for continuum models

For continuum models the underlying stochastic process is a stochastic
differential equation with learnable drift.

The shape of the data is assumed to be [batch, particle, dimension].

The Model class follows the Pytorch-Lightning format.
"""

from argparse import Namespace
import subprocess
import inspect

import torch
import pytorch_lightning as pl

import continuum


DEFAULTS = {
    'git_commit': subprocess.check_output(
        ["git", "describe", "--always"]).strip().decode(),
    ## training parameters
    'epoch_size': 10,
    'batch_size': 1024,
    'lr': 0.01,
    'gamma_lr': 0.95,
    'atol_bias': 0.01,
    ## physical parameters
    'H': 64,
    'method': 'SOSRA',
    'has_vdW': True,
    'strong': False,
    'dt': 0.01,
    'num_steps': 1024,
    ## loss / energy
    'loss_fn': 'boundary_corrected_holland',
    'validate_fn': 'holland_energy',
}

DEFAULT_HPARAMS = Namespace(**DEFAULTS)

ALLOWED_HPARAMS = {'git_commit', 'epoch_size', 'batch_size', 'lr', 'gamma_lr',
                   'H', 'method', 'has_vdW', 'strong', 'dt', 'num_steps',
                   'number_of_particles', 'D', 'net', 'potential', 'R', 'Z',
                   'loss_fn', 'validate_fn', 'atol_bias', 'g', 'omega', 's'}


class Model(pl.LightningModule):
    """ Model for continuum models """

    def __init__(self, hparams):
        """ Returns continuum model.

        Args:
          hparams (Namespace): should contain all parameters needed to setup a model.
            Make hparams by importing default_hparams from this module, and setting
            additional attributes, which should be restricted to the set of allowed_hparams.
        """
        super(Model, self).__init__()
        for key in vars(hparams):
            if key not in ALLOWED_HPARAMS:
                raise AttributeError(f"Hparam {key} is not in ALLOWED_HPARAMS")

        self.hparams = hparams

        self.drift_fn = getattr(continuum.nets, hparams.net)(hparams)

        # Load potential_fn
        self.potential_fn = getattr(continuum.potentials, hparams.potential)
        potential_takes_hparams = 'hparams' in inspect.signature(self.potential_fn).parameters
        if potential_takes_hparams:
            self.potential_fn = self.potential_fn(hparams)

        self.sde = getattr(continuum.sdesolvers, hparams.method)(self.drift_fn, hparams)

        self.state = torch.randn(
            [hparams.batch_size, hparams.number_of_particles, hparams.D]
        )
        self.loss_fn = getattr(continuum.losses, hparams.loss_fn)(self.potential_fn, hparams.has_vdW)
        self.validator = continuum.validate.Validator(self, hparams.atol_bias)

    def forward(self, state, num_steps, dt):
        """ Computes total sde trajectory.

        Args:
          state (Tensor): initial state of sde
          num_steps (int): number of time steps taken for sde
          dt (float): time step of sde

        Returns:
          traj (Tensor): the total sde trajectory.
            If state has shape [batch, particle, dimension], then
            traj has shape [num_steps, batch, particle, dimension]
            and traj[n] contains all positions at time n*dt.
        """
        state_traj = torch.empty((num_steps + 1,) + state.shape, device=state.device)
        drift_traj = torch.empty((num_steps + 1,) + state.shape, device=state.device)
        dW_traj = torch.empty((num_steps,) + state.shape, device=state.device)

        for i in range(num_steps):
            state_traj[i] = state.unsqueeze(dim=0)
            state, drift, dW = self.sde(state, dt)
            drift_traj[i] = drift.unsqueeze(dim=0)
            dW_traj[i] = dW.unsqueeze(dim=0)

        state_traj[-1] = state
        drift_traj[-1] = self.drift_fn(state)

        return state_traj, drift_traj, dW_traj

    def train_dataloader(self):
        return torch.utils.data.DataLoader(ModelState(self, self.hparams.epoch_size))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.sde.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.gamma_lr)

        return [optimizer], [scheduler]

    def training_step(self, state, _):
        """ Computes the training loss. """
        state = state.detach_().squeeze(dim=0)
        state.requires_grad = True

        state_traj, drift_traj, dW_traj = self.forward(state, self.hparams.num_steps, self.hparams.dt)
        self.state = state_traj[-1]

        loss, train_energy = self.loss_fn(state_traj, drift_traj, dW_traj, self.hparams.dt)
        train_log = {'loss': loss, 'train_energy': train_energy}

        return {'loss': loss, 'log': train_log}

    def validation_step(self, state, _):
        """ Computes the validation energy. """
        val_energy = self.validator(state, talks=True)
        return {'val_energy': val_energy}

    def validation_epoch_end(self, outputs):
        """ Averages and logs the validation energy. """
        val_energy_mean = torch.stack([x['val_energy'] for x in outputs]).mean()
        val_log = {'val_energy': val_energy_mean}

        return {'val_energy': val_energy_mean, 'log': val_log}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(ModelState(self, 1))


class ModelState(torch.utils.data.Dataset):
    """ Loads attribute state from argument model.

    The 'data' for the model continuum.Model is a batch of initial states for the sde.
    The model continuum.Model does not load data from disk,
    but these initial states are equal to the final states of the previous episode.
    Therefore, we use a Dataset that simply returns these previous final states.

    Args:
      model (continuum.Model)
      epoch_size (int): number of episodes in one epoch.
    """
    def __init__(self, model, epoch_size):
        super(ModelState, self).__init__()
        self.model = model
        self.epoch_size = epoch_size

    def __getitem__(self, key):
        return self.model.state

    def __len__(self):
        return self.epoch_size
