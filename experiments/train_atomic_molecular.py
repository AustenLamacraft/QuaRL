import sys
from copy import deepcopy
sys.path.append('../../')

import torch
import continuum
from trainer import Trainer


# ## hparams for Helium atom (He)
# hparams = continuum.DEFAULT_HPARAMS
# hparams.net = 'PairDriftHelium'
# hparams.potential = 'he_potential'
# hparams.number_of_particles = 2
# hparams.D = 3
# hparams.lr = 1e-3
# # Train
# model = continuum.Model(hparams)
# trainer = Trainer(name='He', gpus=[3], max_epochs=50)
# trainer.fit(model)


## hparams for Hydrogen atom (H)
# hparams = deepcopy(continuum.DEFAULT_HPARAMS)
# hparams.net = 'DriftResNet'
# hparams.potential = 'h_potential'
# hparams.number_of_particles = 1
# hparams.D = 3
# hparams.H = 256
# hparams.lr = 1e-2
# # Train
# model = continuum.Model(hparams)
# trainer = Trainer(name='H', gpus=[3], max_epochs=50)
# trainer.fit(model)


# hparams for Hydrogen molecule (H2)
hparams = deepcopy(continuum.DEFAULT_HPARAMS)
hparams.net = 'PairDriftH2'
hparams.potential = 'h2_param'
hparams.R = 1.401  # set to 2.8 for wide H2 molecule
hparams.number_of_particles = 2
hparams.D = 3
hparams.H = 64
hparams.lr = 5e-4
# Train
model = continuum.Model(hparams)
trainer = Trainer(name='H2', gpus=[3], max_epochs=50)
torch.cuda.empty_cache()
trainer.fit(model)


# # hparams for wide Hydrogen molecule (H2)
# hparams = deepcopy(continuum.DEFAULT_HPARAMS)
# hparams.net = 'PairDriftH2'
# hparams.potential = 'h2_param'
# hparams.R = 2.8  # set to 2.8 for wide H2 molecule
# hparams.number_of_particles = 2
# hparams.D = 3
# hparams.H = 64
# hparams.lr = 1e-3
# # Train
# model = continuum.Model(hparams)
# trainer = Trainer(name='H2_wide', gpus=[3], max_epochs=100)
# torch.cuda.empty_cache()
# trainer.fit(model)


# Continue training model from checkpoint
checkpoint_path = 'results/H/version_0/_ckpt_epoch_97.ckpt'
model = continuum.Model.load_from_checkpoint(checkpoint_path)
trainer = Trainer(name='H', gpus=[3],
                  version=0, resume_from_checkpoint=checkpoint_path,
                  max_epochs=150)
torch.cuda.empty_cache()
trainer.fit(model)
