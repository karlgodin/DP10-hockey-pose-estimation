from torch import nn
import torch.utils.data
import torch.optim

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import os, sys
from argparse import ArgumentParser


class FPhi(pl.LightningModule):
    def __init__(self, hparams):
        super(FPhi, self).__init__()
        self.hparams = hparams

        if hparams.dataset == "PHYT":
            final_layer_count = 2
        elif hparams.dataset == "SBU":
            final_layer_count = 8

        if hparams.intra:
            first_layer_size = 1000
        elif hparams.inter:
            first_layer_size = 500

        self.model = nn.Sequential(
            nn.Linear(first_layer_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, final_layer_count),
            nn.Softmax(1)
        )

        if self.hparams.full_gpu:
            self.model = self.model.cuda()


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


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parser = ArgumentParser(add_help=False)

    parser = FPhi.add_model_specific_args(parser, root_dir)


    f_model = FPhi(parser.parse_args())
    x = torch.FloatTensor(250)
    f_model(x)
    trainer = Trainer()




