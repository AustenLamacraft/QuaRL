""" Tests for continuum models.
Usage: python -m unittest continuum.test
"""

import os
import unittest
import shutil
from argparse import Namespace

import math
import torch

import continuum
from continuum import Model, DEFAULT_HPARAMS
from continuum.nets import PermEqMean, EquivarDrift, PairDrift, PairDriftH2, PairDriftHelium
from continuum.potentials import h2_param, h2_potential
from continuum.validate import Validator
from trainer import Trainer


class TestPermEqMean(unittest.TestCase):
    """ Test class PermEqMean layer of continuum.nets """
    def test_correctshape(self):

        d_in = 5
        d_out = 10
        num = 4
        batch = 32
        test_net = PermEqMean(d_in, d_out)
        x = torch.randn(batch, num, d_in)
        y = test_net(x)

        self.assertEqual(list(y.shape), [batch, num, d_out])

    def test_isequivariant(self):

        d_in = 5
        d_out = 10
        num = 4
        batch = 32
        test_net = PermEqMean(d_in, d_out)
        x = torch.randn(batch, num, d_in)
        y = test_net(x)

        perm = torch.randperm(num)
        x_perm = x[:, perm, :]
        y_perm = y[:, perm, :]

        self.assertTrue(torch.allclose(y_perm, test_net(x_perm), rtol=4e-05, atol=4e-08,))


class TestEquivarDrift(unittest.TestCase):
    """ Test class EquivarDrift of continuum.nets """
    def test_correctshape(self):

        d = 5
        num_channels = 10
        num = 4
        batch = 32
        test_net = EquivarDrift(d, num_channels)
        x = torch.randn(batch, num, d)
        y = test_net(x)

        self.assertEqual(list(y.shape), [batch, num, d])

    def test_isequivariant(self):

        d = 5
        num_channels = 10
        num = 4
        batch = 32
        test_net = EquivarDrift(d, num_channels)
        x = torch.randn(batch, num, d)
        y = test_net(x)

        perm = torch.randperm(num)
        x_perm = x[:, perm, :]
        y_perm = y[:, perm, :]

        self.assertTrue(torch.allclose(y_perm, test_net(x_perm)))

class TestPairDrift(unittest.TestCase):
    """ Test class PairDrift of continuum.nets """
    def setUp(self):
        d = 5
        num_channels = 10
        number_of_particles = 4
        batch_size = 32
        hparams = Namespace(D=d, H=num_channels)
        self.net = PairDrift(hparams)
        self.shape = (batch_size, number_of_particles, d)
        self.x = torch.randn(self.shape)
        self.y = self.net(self.x)

    def test_correctshape(self):
        self.assertEqual(self.y.shape, self.shape)

    def test_isequivariant(self):
        perm = torch.randperm(self.shape[1])
        x_perm = self.x[:, perm, :]
        y_perm = self.y[:, perm, :]

        self.assertTrue(torch.allclose(y_perm, self.net(x_perm)))


class TestPairDriftH2(unittest.TestCase):
    """ Test class PairDriftH2 of continuum.nets """
    def setUp(self):
        d = 3
        num_channels = 10
        number_of_particles = 2
        batch_size = 32
        hparams = Namespace(D=d, H=num_channels, R=1.0)
        self.net = PairDriftH2(hparams)
        self.shape = (batch_size, number_of_particles, d)
        self.x = torch.randn(self.shape)
        self.y = self.net(self.x)

    def test_correctshape(self):
        self.assertEqual(self.y.shape, self.shape)

    def test_isequivariant(self):
        perm = torch.randperm(self.shape[1])
        x_perm = self.x[:, perm, :]
        y_perm = self.y[:, perm, :]

        self.assertTrue(torch.allclose(y_perm, self.net(x_perm)))


class TestPairDriftHelium(unittest.TestCase):
    """ Test class PairDriftHelium of continuum.nets """
    def setUp(self):
        d = 3
        num_channels = 10
        number_of_particles = 2
        batch_size = 32
        hparams = Namespace(D=d, H=num_channels)
        self.net = PairDriftHelium(hparams)
        self.shape = (batch_size, number_of_particles, d)
        self.x = torch.randn(self.shape)
        self.y = self.net(self.x)

    def test_correctshape(self):
        self.assertEqual(self.y.shape, self.shape)

    def test_isequivariant(self):
        perm = torch.randperm(self.shape[1])
        x_perm = self.x[:, perm, :]
        y_perm = self.y[:, perm, :]

        self.assertTrue(torch.allclose(y_perm, self.net(x_perm)))
        

class TestH2Param(unittest.TestCase):
    """ Test method h2_param of continuum.potentials. """
    def test_matchesH2potential(self):

        particles = 4
        d = 3
        batch = 32

        x = torch.randn(batch, particles, d)

        self.assertTrue(torch.allclose(h2_param({})(x), h2_potential(x)))


class TestSHOEnergy(unittest.TestCase):
    """ Test costs given by Model and Validator (continuum.validate)
    using the simple harmonic oscillator."""
    @staticmethod
    def sho_drift(x):
        return -x

    @staticmethod
    def sho_potential(x):
        return 0.5 * torch.sum(x**2, dim=(1, 2))

    @staticmethod
    def sho_stationary_state(batch_size, particles, d):
        return torch.randn(batch_size, particles, d) / math.sqrt(2)

    def sho_energy_test(self, solver, state=None, atol=0.05,
                        particles=1, d=3, num_steps=1024, dt=0.01):
        if state is None:
            state = self.sho_stationary_state(8192, 1, 3)

        with torch.no_grad():
            state_traj, drift_traj, dW_traj = solver(state, num_steps, dt)
            loss_fn = continuum.losses.holland(solver.potential_fn)
            avg_cost, _ = loss_fn(state_traj, drift_traj, dW_traj, dt)

        if not torch.allclose(avg_cost, torch.tensor(0.5*d*particles), atol=atol):
            print(f"test_SHO_Energy: {avg_cost:.3f} is not equal to"
                  f"{0.5*d*particles:.3f}")
            self.assertTrue(torch.allclose(
                avg_cost, torch.tensor(0.5*d*particles), atol=atol))

    def test_holland_cost(self):
        for method in ['Euler', 'Heun', 'SRA3', 'SOSRA']:
            for strong in [True, False]:
                for has_vdW in [True, False]:
                    hparams = DEFAULT_HPARAMS
                    hparams.net = 'DriftResNet'
                    hparams.number_of_particles = 1
                    hparams.potential = 'sho_potential'
                    hparams.D = 3
                    hparams.method = method
                    hparams.strong = strong
                    hparams.has_vdW = has_vdW

                    solver = Model(hparams)
                    self.sho_energy_test(solver, num_steps=64)

    def test_isfinite_boundary_corrected_holland_cost(self):
        hparams = DEFAULT_HPARAMS
        hparams.net = 'DriftResNet'
        hparams.number_of_particles = 1
        hparams.potential = 'sho_potential'
        hparams.D = 3

        solver = Model(hparams)

        with torch.no_grad():
            state = self.sho_stationary_state(8192, 1, 3)
            state_traj, drift_traj, dW_traj = solver(state, 64, 0.01)
            loss_fn = continuum.losses.boundary_corrected_holland(solver.potential_fn)
            avg_cost, _ = loss_fn(state_traj, drift_traj, dW_traj, 0.01)
            self.assertTrue(torch.isfinite(avg_cost))

    def test_evaluated_SHO_energy(self):
        h = DEFAULT_HPARAMS
        h.number_of_particles, h.D = 1, 3
        h.potential = 'sho_potential'
        h.net = 'HarmonicNet'
        h.Z = 1.0
        atol = 0.05

        model = continuum.Model(h)

        with torch.no_grad():
            validator = Validator(model, atol)
            state = self.sho_stationary_state(h.batch_size, h.number_of_particles, h.D)
            val_energy = validator(state)

        exact_energy = torch.tensor(0.5 * h.D * h.number_of_particles)
        if not torch.allclose(val_energy, exact_energy, atol=atol, rtol=0):
            print(f'test_evaluated_SHO_Energy: {val_energy:.3f}+-{atol:.3f} is not'
                  f'within {atol:.3f} of {exact_energy:.3f}')
        self.assertTrue(torch.allclose(val_energy, exact_energy, atol=atol))


class TestTraining(unittest.TestCase):
    """ Test PyTorch-Lightning Training routine.
    Does the pre-training routine, which includes a validation
    sanity check, as well as one training step. """

    def setUp(self):
        self.hparams = continuum.DEFAULT_HPARAMS
        self.hparams.net = 'PeriodicCNN'
        self.hparams.net = 'DriftResNet'
        self.hparams.potential = 'sho_potential'
        self.hparams.number_of_particles = 1
        self.hparams.D = 3
        self.hparams.H = 256
        self.hparams.num_steps = 16
        self.hparams.atol_bias = 10.0
        self.hparams.epoch_size = 1

    def test_no_exceptions_sho(self):
        # Make hparams from default_hparams
        model = continuum.Model(self.hparams)
        trainer = Trainer(name='test', max_epochs=1)
        trainer.fit(model)

    def tearDown(self):
        if os.path.exists('results/test'):
            shutil.rmtree('results/test')
        if os.path.exists('results'):
            os.rmdir('results')  # if empty
