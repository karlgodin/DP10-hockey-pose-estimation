from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim

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
        #print(f"[{datetime.datetime.now()}] Hyperparameters\n{hparams}")
        self.hparams = hparams

        # define model architecture
        # where 1000 is the number of frames * 3(x, y, c) * 2 (2 joints)
        nb_nodes = 745 #560
        self.model = nn.Sequential(
            nn.Linear(nb_nodes, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, 250),
            nn.Softmax(dim=0)
        )

        if self.hparams.full_gpu:
            self.model = self.model.cuda()

        #print(self.model[0], self.model[0].weight)

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

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])

        # Specify whether or not to put entire dataset on GPU
        parser.add_argument('--full_gpu', default=True, type=bool)

        # training params (opt)
        parser.add_argument('--patience', default=10, type=int)
        parser.add_argument('--optim', default='Adam', type=str)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--momentum', default=0.0, type=float)
        parser.add_argument('--nesterov', default=False, type=bool)
        parser.add_argument('--batch_size', default=32, type=int)
        return parser


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

        # early_stop_callback = EarlyStopping(
        #     monitor='val_acc',
        #     min_delta=0.00,
        #     patience=hparams.patience,
        #     verbose=False,
        #     mode='max'
        # )

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

# have the pairs of joints together to feed in the network
def get_combinations(perp: torch.FloatTensor, victim: torch.FloatTensor):
    sizeOfPerp = perp.size()
    nb_position_input = sizeOfPerp[2]
    array_body_index = np.arange(25)
    nb_frames = int(nb_position_input / 3)

    values = itertools.product(array_body_index, repeat=2)
    nb_players = 2
    sizeOfData = nb_frames * 3 * nb_players + nb_frames + (nb_frames - 1) + nb_players
    outputs = []
    for x in values:
        person1 = perp[0]
        person2 = victim[0]
        joint1 = person1[x[0]]
        joint2 = person2[x[1]]

        # get features for distance and motion
        distances = DP10_getDistanceVector2Poses(joint1, joint2)
        distances = torch.tensor(distances, dtype=torch.float32)
        motions = DP10_getMotionVector2Poses(joint1, joint2)
        motions = torch.tensor(motions, dtype=torch.float32)
        # put on the same row for the matrix: joint1, joint2, distances, motions
        iteration1 = torch.cat([joint1, joint2, distances, motions], dim=0)
        outputs.append(iteration1)

    outputs = torch.stack(outputs, dim=0)
    return outputs

if __name__ == '__main__':
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = ArgumentParser(add_help=False)

    # get the inputs
    perp, victim = parse_clip()
    inter_combinations_joints = get_combinations(perp, victim)
    # allow model to overwrite or extend args
    parser = GTheta.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # train model
    main(hyperparams, None)
