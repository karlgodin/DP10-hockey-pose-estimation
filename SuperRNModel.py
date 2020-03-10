import os, sys
from argparse import ArgumentParser
from classifier.joints import parse_clip
from classifier.GTheta import get_combinations
from classifier.GTheta import GTheta
from classifier.FPhi import FPhi
from classifier.dataset import PHYTDataset, SBUDataset
import torch.tensor
import torch.nn as nn

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import math
import numpy as np

from pytorch_lightning.callbacks import EarlyStopping

def accuracy(y_pred, y):
    correct = (y == y_pred).sum().float()
    acc = correct/len(y)
    return acc


class SuperRNModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SuperRNModel, self).__init__()
        self.hparams = hparams

        self.g_model = GTheta(hparams)
        self.f_model = FPhi(hparams)

        dataset = SBUDataset('classifier/SBUDataset')
        self.criterion = nn.CrossEntropyLoss()

        [training_dataset, validation_dataset] = torch.utils.data.random_split(dataset, [round(len(dataset) * 0.9), round(len(dataset) * 0.1)])
        self.train_dataset = training_dataset
        self.val_dataset = validation_dataset




    def forward(self, x):
        # numpy matrix of all combination of inter joints

        perp = x[:,:int(x.shape[1]/2),:]
        victim = x[:,int(x.shape[1]/2):,:]

        input_data_clip_combinations = get_combinations(perp, victim)
        tensor_g = self.g_model(input_data_clip_combinations)

        # calculate sum and div
        sum = torch.sum(tensor_g, dim=1)
        size_output_G = tensor_g.shape[1]
        average_output = sum / size_output_G

        tensor_classification = self.f_model(average_output)
        return tensor_classification

    def configure_optimizers(self):
        if self.hparams.optim == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optim == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        else:
            return None

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x,y = batch
        if self.hparams.full_gpu:
            x = x.cuda()
            y = y.cuda()
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, torch.max(y, 1)[1])
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        if self.hparams.full_gpu:
            x = x.cuda()
            y = y.cuda()
        y_hat = self.forward(x)
        return {'val_loss': self.criterion(y_hat, torch.max(y, 1)[1]), 'val_acc': accuracy(torch.argmax(y_hat, 1), torch.max(y, 1)[1])}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'log': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False)


if __name__ == '__main__':
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = ArgumentParser(add_help=False)

    # get the inputs for the black box(all the clips)
    # parse each clip to its joints
    #perp, victim = parse_clip()

    # allow model to overwrite or extend args
    parser = GTheta.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    rnModel = SuperRNModel(hyperparams)

    for name, params in rnModel.named_parameters():
        print(name, '\t\t', params.shape)

    print('\n')

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=hyperparams.patience,
        verbose=False,
        mode='max'
    )

    trainer = Trainer(max_nb_epochs=200, early_stop_callback=early_stop_callback, checkpoint_callback=None)

    trainer.fit(rnModel)




