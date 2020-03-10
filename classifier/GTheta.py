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
    def __init__(self, hparams,numOfFrames,numOfJoints):
        super(GTheta, self).__init__()
        #print(f"[{datetime.datetime.now()}] Hyperparameters\n{hparams}")
        self.hparams = hparams

        # define model architecture
        # where 1000 is the number of frames * 3(x, y, c) * 2 (2 joints)
        nb_nodes =  numOfFrames*8 + 1#Input size is #ofFrames*8+1
        self.model = nn.Sequential(
            nn.Linear(nb_nodes, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, nb_nodes),
            nn.ReLU(),
            nn.Linear(nb_nodes, 250)
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

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False)

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
def get_combinations_inter(perp: torch.FloatTensor, victim: torch.FloatTensor):
    sizeOfPerp = perp.size()
    nb_position_input = sizeOfPerp[2]
    array_body_index = np.arange(25)
    nb_frames = int(nb_position_input / 3)

    values = itertools.product(array_body_index, repeat=2)
    outputs = []
    for x in values:
        person1 = perp[0]
        person2 = victim[0]
        joint1 = person1[x[0]]
        joint2 = person2[x[1]]

        x_1 = joint1[::3]
        x_1 = x_1[:-1]
        x_2 = joint2[::3]
        x_2 = x_2[:-1]

        y_1 = joint1[1::3]
        y_2 = joint2[1::3]

        distances = torch.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)

        motions = torch.sqrt((x_1[:-1] - x_2[1:])**2 + (y_1[:-1] - y_2[1:])**2 )

        # put on the same row for the matrix: joint1, joint2, distances, motions
        iteration1 = torch.cat([joint1, joint2, distances, motions], dim=0)
        outputs.append(iteration1)
    
    #Shape is [(#ofJoints)**2, #ofFrames*8+1]
    outputs = torch.stack(outputs, dim=0)
    return outputs
    
# have the pairs of joints together to feed in the network
def get_combinations_intra(p1: torch.FloatTensor):
    sizeOfP1 = p1.size()
    nb_position_input = sizeOfP1[2]
    array_body_index = np.arange(25)
    nb_frames = int(nb_position_input / 3)

    values = itertools.combinations(array_body_index, 2)
    outputs = []
    for x in values:
        person1 = p1[0]
        joint1 = person1[x[0]]
        joint2 = person1[x[1]]

        x_1 = joint1[::3]
        x_1 = x_1[:-1]
        x_2 = joint2[::3]
        x_2 = x_2[:-1]

        y_1 = joint1[1::3]
        y_2 = joint2[1::3]

        distances = torch.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)

        motions = torch.sqrt((x_1[:-1] - x_2[1:])**2 + (y_1[:-1] - y_2[1:])**2 )
        

        # put on the same row for the matrix: joint1, joint2, distances, motions
        iteration1 = torch.cat([joint1, joint2, distances, motions], dim=0)
        outputs.append(iteration1)
    
    #Shape is [(#ofJoints)!/2!/(#ofJoints - 2)!, #ofFrames*8+1]
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
