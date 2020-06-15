import sys
sys.path.append('../../')
import continuum
import torch


paths = [
    'results/H/version_34/_ckpt_epoch_97.ckpt',
    'results/He/version_0/_ckpt_epoch_49.ckpt',
    ]


for checkpoint_path in paths:
    model = continuum.Model.load_from_checkpoint(checkpoint_path)

    device = 'cuda:3'
    model = model.to(device=device)
    model.state = model.state.to(device=device)
    model.drift_fn = model.drift_fn.to(device=device)

    atol_bias = 0.00001
    validator = continuum.validate.Validator(model, atol_bias)

    with torch.no_grad():
        validator(model.state, talks=True, max_stages=8)
