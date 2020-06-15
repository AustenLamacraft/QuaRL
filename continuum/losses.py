""" Module for loss functions. """
import torch
import pickle


def holland(potential_fn, has_vdW=True):
    """ Calculates the log-likelihood ell of a trajectory.
    Uses left Riemann sum for integration. """

    def loss_fn(state_traj, drift_traj, dW_traj, dt):
        potential = potential_fn(state_traj[:-1])

        kinetic = 0.5 * torch.sum(drift_traj[:-1]**2, dim=(-1, -2))

        if has_vdW:
            kinetic += torch.sum(drift_traj[:-1] * dW_traj / dt, dim=(-1, -2))
        
        loss = (potential + kinetic).mean()  # average over time and batch
        return loss, loss

    return loss_fn
    

def boundary_corrected_holland(potential_fn, has_vdW=True):
    """ Calculates the log-likelihood ell of a trajectory.
    Uses left Riemann sum for integration.
    Corrects for the boundary term in the RN. """

    def loss_fn(state_traj, drift_traj, dW_traj, dt):
        potential = potential_fn(state_traj[:-1])

        kinetic = 0.5 * torch.sum(drift_traj[:-1]**2, dim=(-1, -2))

        if has_vdW:
            kinetic += torch.sum(drift_traj[:-1] * dW_traj / dt, dim=(-1, -2))
        
        drift_dot_state = torch.sum(drift_traj[-1].detach() * state_traj[-1], dim=(-1, -2))

        energy = (potential + kinetic).mean()  # average over time and batch
        T = ((state_traj.shape[0] - 1) * dt)  # T = num_steps * dt
        loss = energy - drift_dot_state.mean() / T
        return loss, energy

    return loss_fn
