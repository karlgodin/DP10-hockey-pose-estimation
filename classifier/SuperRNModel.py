import os, sys
from argparse import ArgumentParser
from classifier.joints import parse_clip
from classifier.GTheta import get_combinations
from classifier.GTheta import GTheta
from classifier.FPhi import FPhi
from classifier.dataset import PHYTDataset
import torch.tensor

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader


import numpy as np

from pytorch_lightning.callbacks import EarlyStopping


class SuperRNModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SuperRNModel, self).__init__()
        self.hparams = hparams

        self.g_model = GTheta(hparams)
        self.f_model = FPhi(hparams)

        self.dataset = PHYTDataset('clips/')


    def forward(self, x):
        print("Begin: forwarding of the SuperRNModel")

        print("starting calculations for G function")
        # numpy matrix of all combination of inter joints

        perp = x['perp']
        victim = x['victim']

        input_data_clip_combinations = get_combinations(perp, victim)
        tensor_g = self.g_model(input_data_clip_combinations)

        # calculate sum and div
        sum = torch.sum(tensor_g, dim=0)
        size_output_G = tensor_g.shape[1]
        average_output = sum / size_output_G

        print("starting forward for F function")
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
        x = batch
        if not self.hparams.full_gpu:
            x = x.cuda()
            y = y.cuda()
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.squeeze())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True)

    # pseudo code for our implementation of the rn model
    # master network:
    #     forward():
    #         for i in 625:
    #             accu += gtheta.forward(x)
    #
    #         x = accu/625
    #
    #         return = fphi.forward(x)


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

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=hyperparams.patience,
        verbose=False,
        mode='max'
    )

    trainer = Trainer(max_nb_epochs=1, early_stop_callback=early_stop_callback, checkpoint_callback=None)

    trainer.fit(rnModel)




