from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim
from classifier.dataset import target_frame_amount
import itertools

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TestTubeLogger

from classifier.joints import parse_clip
from datasetCreation.scripts.DP10_DataAugmentation import DP10_getDistanceVector2Poses, DP10_getMotionVector2Poses
import numpy as np

import os, sys
from argparse import ArgumentParser
import datetime

class GTheta(pl.LightningModule):
    def __init__(self, hparams):
        super(GTheta, self).__init__()
        self.hparams = hparams

        # define model architecture
        # where 1000 is the number of frames * 3(x, y, c) * 2 (2 joints)
        nb_nodes = target_frame_amount * 3 * 2 + 2 + target_frame_amount * 2 - 1
        self.model = nn.Sequential(
            nn.Linear(nb_nodes, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.Dropout(0.1)
        )

        if self.hparams.full_gpu:
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)

    # configure the optimizer, from paper, use Adam and lr = 0.0001
    def configure_optimizers(self):
        if self.hparams.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            return [optimizer], []
        elif self.hparams.optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
            return [optimizer], []
        else:
            raise ValueError(f'[ERROR] Invalid optimizer {self.hparams.optim}')




def main(hparams, version=None):
    torch.manual_seed(0)

    if version:
        tt_logger = TestTubeLogger(
            save_dir="logs",
            name="relational_model",
            debug=False,
            create_git_tag=False,
            version=version
        )
    else:
        tt_logger = TestTubeLogger(
            save_dir="logs",
            name="relational_model",
            debug=False,
            create_git_tag=False
        )

    checkpoint_callback = ModelCheckpoint(
        filepath=f'logs/{tt_logger.name}/version_{tt_logger.experiment.version}/checkpoints',
        save_best_only=True,
        verbose=False,
        monitor='val_loss',
        mode='max',
        prefix=''
    )

    model = GTheta(hparams)
    trainer = Trainer(max_nb_epochs=100, early_stop_callback=None, checkpoint_callback=checkpoint_callback,
                      logger=tt_logger, gpus=[0], log_save_interval=400)
    trainer.fit(model)
    # trainer.test(model)


if __name__ == '__main__':
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = ArgumentParser(add_help=False)

    # get the inputs
    perp, victim = parse_clip()
    inter_combinations_joints = get_combinations_inter(perp, victim)
    # allow model to overwrite or extend args
    parser = GTheta.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams, None)
