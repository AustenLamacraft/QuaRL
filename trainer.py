""" Sets defaults for the Pytorch-Lightning (pl) Trainer. """

import os
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class Trainer(pl.Trainer):
    """ Sets defaults for the Pytorch-Lightning (pl) Trainer.

    Any keyword arguments are passed directly to pl.Trainer,
    and override the defaults defined in this module.
    """

    def __init__(self, **kwargs):

        # Experiment results of name 'foo' are placed in directory results/foo/version_n/
        kwargs.setdefault('logger', loggers.TensorBoardLogger(
            'results/', name=kwargs['name'], version=kwargs.get('version')))

        # Early stopping is disabled
        kwargs.setdefault('early_stop_callback', False)

        # Create results and/or results/name if they don't exist
        if not os.path.exists('results'):
            os.system('mkdir results')
        if not os.path.exists('results/' + kwargs['name']):
            os.system('mkdir results/' + kwargs['name'])

        # Checkpoint are saved in directory results/foo/version_n/
        kwargs.setdefault('checkpoint_callback', ModelCheckpoint(
            filepath=('results/' + kwargs['name'] + '/version_'
                      + str(kwargs['logger'].version) + '/c'),
            monitor='val_energy',
            prefix='',
            save_top_k=5
        ))

        kwargs.setdefault('log_save_interval', 100)  # logs are written to disk every 100 episodes
        kwargs.setdefault('row_log_interval', 1)  # logs are created every episode
        kwargs.setdefault('progress_bar_refresh_rate', 1)

        super(Trainer, self).__init__(**kwargs)
