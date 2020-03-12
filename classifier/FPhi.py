from torch import nn
from pytorch_lightning import Trainer
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TestTubeLogger

import os, sys
from argparse import ArgumentParser
import datetime


class FPhi(pl.LightningModule):
    def __init__(self, hparams,sizeOfInputNode):
        super(FPhi, self).__init__()
        self.hparams = hparams

        nb_nodes =  sizeOfInputNode#Input size is #ofJoints**2
        self.model = nn.Sequential(
            nn.Linear(nb_nodes, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
            nn.Softmax(0)
        )

        if self.hparams.full_gpu:
            self.model = self.model.cuda()

        #print(self.model[0], self.model[0].weight)

    def forward(self, x):
        return self.model(x)

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

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])

        # Specify whether or not to put entire dataset on GPU
        parser.add_argument('--full_gpu', default=False, type=bool)

        # Threshold for data "preprocessing"
        parser.add_argument('--threshold', default=254, type=int)

        # training params (opt)
        parser.add_argument('--patience', default=5, type=int)
        parser.add_argument('--optim', default='Adam', type=str)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--momentum', default=0.0, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        return parser


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parser = ArgumentParser(add_help=False)

    parser = FPhi.add_model_specific_args(parser, root_dir)


    f_model = FPhi(parser.parse_args())
    x = torch.FloatTensor(250)
    f_model(x)
    trainer = Trainer()

    #trainer.fit(f_model)



    #parser = FPhi.add_model_specific_args(parent_parser, root_dir)
    #hyperparams = parser.parse_args()

    # train model
    #main(hyperparams, None)

"""
master network:
    forward():
        for i in 625:
            accu += gtheta.forward(x)

        x = accu/625

        return = fphi.forward(x)

"""

"""
class GTheta(pl.LightningModule):
    def __init__(self, hparams):
        super(GTheta, self).__init__()
        print(f"[{datetime.datetime.now()}] Hyperparameters\n{hparams}")
        self.hparams = hparams

        # define model architecture
        # where 1000 is the number of frames * 3(x, y, c) * 2 (2 joints)
        self.model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.Sigmoid()
        )
        print(self.model[0], self.model[0].weight)

        # recuperer le dataset quon vient de preprocess here. Joints
        dataset = []

        [training_dataset, validation_dataset] = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9),
                                                                                         int(len(dataset) * 0.1)])

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

"""




