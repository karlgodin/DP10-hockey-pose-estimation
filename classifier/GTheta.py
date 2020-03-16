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



# have the pairs of joints together to feed in the network
def get_combinations_inter(perps: torch.FloatTensor, victims: torch.FloatTensor):
    num_distances = int((perps.size()[2] - 1) / 3)

    outputs = torch.empty(perps.size()[0], perps.size()[1] * perps.size()[1],
                          perps.size()[2] * 2 + 2 * num_distances - 1)

    for count, perp in enumerate(perps):
        victim = victims[count]
        sizeOfPerp = perp.size()
        nb_position_input = sizeOfPerp[1]
        array_body_index = np.arange(sizeOfPerp[0])
        nb_frames = int((nb_position_input - 1) / 3)

        values = itertools.product(array_body_index, repeat=2)
        nb_players = 2
        # TODO transform this into good comment that describes size of data: sizeOfData = nb_frames * 3 * nb_players + nb_frames + (nb_frames - 1) + nb_players
        output = []
        for x in values:
            joint1 = perp[x[0]]
            joint2 = victim[x[1]]

            x_1 = joint1[::3]
            x_1 = x_1[:-1]
            x_2 = joint2[::3]
            x_2 = x_2[:-1]

            y_1 = joint1[1::3]
            y_2 = joint2[1::3]

            distances = torch.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

            motions = torch.sqrt((x_1[:-1] - x_2[1:]) ** 2 + (y_1[:-1] - y_2[1:]) ** 2)

            # put on the same row for the matrix: joint1, joint2, distances, motions
            iteration1 = torch.cat([joint1, joint2, distances, motions], dim=0)
            output.append(iteration1)

        output = torch.stack(output, dim=0)
        outputs[count] = output
    return outputs


def get_combinations_intra(p1: torch.FloatTensor):
    num_distances = int((p1.size()[2] - 1) / 3)

    outputs = torch.empty(p1.size()[0], int((p1.size()[1] * p1.size()[1] - p1.size()[1])/2),
                          p1.size()[2] * 2 + 2 * num_distances - 1)

    for count, player1 in enumerate(p1):
        sizeOfPlayer1 = player1.size()
        nb_position_input = sizeOfPlayer1[1]
        array_body_index = np.arange(sizeOfPlayer1[0])
        nb_frames = int((nb_position_input - 1) / 3)

        values = itertools.combinations(array_body_index, 2)
        nb_players = 2
        # TODO transform this into good comment that describes size of data: sizeOfData = nb_frames * 3 * nb_players + nb_frames + (nb_frames - 1) + nb_players
        output = []
        for x in values:
            joint1 = player1[x[0]]
            joint2 = player1[x[1]]

            x_1 = joint1[::3]
            x_1 = x_1[:-1]
            x_2 = joint2[::3]
            x_2 = x_2[:-1]

            y_1 = joint1[1::3]
            y_2 = joint2[1::3]

            distances = torch.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

            motions = torch.sqrt((x_1[:-1] - x_2[1:]) ** 2 + (y_1[:-1] - y_2[1:]) ** 2)

            # put on the same row for the matrix: joint1, joint2, distances, motions
            iteration1 = torch.cat([joint1, joint2, distances, motions], dim=0)
            output.append(iteration1)

        output = torch.stack(output, dim=0)
        outputs[count] = output
    return outputs


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
