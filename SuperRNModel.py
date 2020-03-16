import os, sys
from argparse import ArgumentParser
from classifier.joints import parse_clip
from classifier.GTheta import get_combinations
from classifier.GTheta import GTheta
from classifier.FPhi import FPhi
from classifier.dataset import PHYTDataset
import torch.tensor
import torch.nn as nn

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from test_tube.hpc import SlurmCluster

import numpy as np

from pytorch_lightning.callbacks import EarlyStopping


class SuperRNModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SuperRNModel, self).__init__()
        self.hparams = hparams

        self.g_model = GTheta(hparams)
        self.f_model = FPhi(hparams)

        self.dataset = PHYTDataset('classifier/clips/')
        self.criterion = nn.BCELoss()


    def forward(self, x):
        # numpy matrix of all combination of inter joints

        perp = x[:,:25,:]
        victim = x[:,25:,:]

        input_data_clip_combinations = get_combinations(perp, victim)
        tensor_g = self.g_model(input_data_clip_combinations)

        # calculate sum and div
        sum = torch.sum(tensor_g, dim=0)
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

def train(hyperparams):
    rnModel = SuperRNModel(hyperparams)
    print("hyperparameters")
    print(hyperparams)

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

    trainer = Trainer(max_nb_epochs=1, early_stop_callback=early_stop_callback, checkpoint_callback=None)

    trainer.fit(rnModel)

if __name__ == '__main__':
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = ArgumentParser(add_help=False)
    # add the grid search

    # get the inputs for the black box(all the clips)
    # parse each clip to its joints
    #perp, victim = parse_clip()
    parser = GTheta.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    # could optimize the number of layers and nodes (parser.opt_list, parser.opt_range)

    # train
    train(hyperparams)

    # rnModel = SuperRNModel(hyperparams)
    # print("hyperparameters")
    # print(hyperparams)
    #
    # for name, params in rnModel.named_parameters():
    #     print(name, '\t\t', params.shape)
    #
    # print('\n')
    #
    # early_stop_callback = EarlyStopping(
    #     monitor='val_acc',
    #     min_delta=0.00,
    #     patience=hyperparams.patience,
    #     verbose=False,
    #     mode='max'
    # )
    #
    # trainer = Trainer(max_nb_epochs=1, early_stop_callback=early_stop_callback, checkpoint_callback=None)
    #
    # trainer.fit(rnModel)

    # do grid search here
    # subclass of argparse


    # cluster = SlurmCluster(
    #     hyperparam_optimizer=hyperparams,
    #     log_path='/path/to/log/results/to',
    #     python_cmd='python2'
    # )
    #
    # # let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
    # cluster.notify_job_status(email='marine.huynh@yahoo.ca', on_done=True, on_fail=True)
    # # set the job options. In this instance, we'll run 20 different models
    # # each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
    # cluster.per_experiment_nb_gpus = 1
    # cluster.per_experiment_nb_nodes = 1
    # # we'll request 10GB of memory per node
    # cluster.memory_mb_per_node = 100
    # # set a walltime of 10 minues
    # cluster.job_time = '10:00'
    #
    # # run the models on the cluster
    # cluster.optimize_parallel_cluster_gpu(
    #     train_model_for_main,
    #     nb_trials=20,
    #     job_name='my_grid_search_exp_name',
    #     job_display_name='my_exp')

    # the function seem to not work, why? would be straight forward
    value = hyperparams.optimize_parallel_cpu(train, nb_trials=20, nb_workers=1)
    #value = hyperparams.optimize_parallel_gpu(train, gpu_ids=['0', '1'])
    print(value)

