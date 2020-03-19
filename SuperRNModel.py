import os, sys
from argparse import ArgumentParser
from classifier.joints import parse_clip

from classifier.GTheta import get_combinations_inter, get_combinations_intra
from classifier.GTheta import GTheta
from classifier.FPhi import FPhi
from classifier.dataset import PHYTDataset, generatorKFold, SBUDataset
import classifier.dataset
import torch.tensor
import torch.nn as nn

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import numpy as np
import math

def accuracy(y_pred, y):
    correct = (y == y_pred).sum().float()
    acc = correct/len(y)
    return acc


from pytorch_lightning.callbacks import EarlyStopping

def add_model_specific_args(parent_parser, root_dir):
    parser = ArgumentParser(parents=[parent_parser])

    # Specify whether or not to put entire dataset on GPU
    parser.add_argument('--full_gpu', action='store_true')

    # training params (opt)
    parser.add_argument('--patience', default=4, type=int)
    parser.add_argument('--kfold', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--changeOrder', action='store_true')
    parser.add_argument('--randomJointOrder', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--inter', action='store_true')
    parser.add_argument('--intra', action='store_true')
    parser.add_argument('--dataset', default='PHYT', type=str)
    return parser


class SuperRNModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SuperRNModel, self).__init__()
        self.hparams = hparams

        self.isInter = self.hparams.inter and not self.hparams.intra
        self.isIntra = not self.hparams.inter and self.hparams.intra

        print('We are using:',hparams.dataset)
        if hparams.dataset == "PHYT":
            self.criterion = nn.BCELoss()
            self.dataset = PHYTDataset('datasetCreation/FilteredPoses/',hparams)
        elif hparams.dataset == "SBU":
            self.criterion = nn.CrossEntropyLoss()
            self.dataset = SBUDataset('classifier/SBUDataset', hparams)

        #Load dataset and find its size
        numOfFrames = (len(self.dataset.clips[0][0])-1)//3
        numOfJoints = len(self.dataset.clips[0])//2
        
        if(self.isInter):
            #Init Gmodel
            self.g_model_inter = GTheta(hparams)
            #Init FModel
            self.f_model = FPhi(hparams)
        
        if(self.isIntra):
            #Init Gmodel
            self.g_model_intra = GTheta(hparams)
            
            #Init FModel
            sizeFInput = 2 * math.factorial(numOfJoints)//math.factorial(2)//math.factorial(numOfJoints - 2)
            self.f_model = FPhi(hparams)

    def forward(self, x):
        # numpy matrix of all combination of inter joints

        p1 = x[:,:int(x.shape[1]/2),:]
        p2 = x[:,int(x.shape[1]/2):,:]
        
        if(self.isInter):
            input_data_clip_combinations = get_combinations_inter(p1, p2)
            
            if(self.hparams.full_gpu):
                input_data_clip_combinations = input_data_clip_combinations.cuda()
            
            tensor_g = self.g_model_inter(input_data_clip_combinations)

            # calculate sum and div
            sum = torch.sum(tensor_g, dim=1)
            size_output_G = tensor_g.shape[1]
            average_output = sum / size_output_G

            tensor_classification = self.f_model(average_output)
            
        if(self.isIntra):
            input_data_clip_combinations_P1 = get_combinations_intra(p1)
            input_data_clip_combinations_P2 = get_combinations_intra(p2)
            
            if(self.hparams.full_gpu):
                input_data_clip_combinations_P1 = input_data_clip_combinations_P1.cuda()
                input_data_clip_combinations_P2 = input_data_clip_combinations_P2.cuda()
                
            tensor_g_P1 = self.g_model_intra(input_data_clip_combinations_P1)
            tensor_g_P2 = self.g_model_intra(input_data_clip_combinations_P2)

            # calculate sum and div
            average_output = torch.empty((0))
            if(self.hparams.full_gpu):
                average_output = average_output.cuda()
            
            for tensor_g in [tensor_g_P1,tensor_g_P1]:
                sum = torch.sum(tensor_g, dim=1)
                size_output_G = tensor_g.shape[1]
                average_output_temp = sum / size_output_G
                average_output = torch.cat([average_output, average_output_temp], dim=1)
                
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
        if self.hparams.dataset == "PHYT":
            loss = self.criterion(y_hat.squeeze(), y.squeeze())
        elif self.hparams.dataset == "SBU":
            loss = self.criterion(y_hat, torch.max(y, 1)[1])

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True)


    @pl.data_loader
    def val_dataloader(self):
        self.valResults = []
        class temp():
            def __init__(self,x,y):
                self.x = x
                self.y = y
            
            def __len__(self):
                return len(self.x)
            def __getitem__(self, index):
                return self.x[index], self.y[index]
        return DataLoader(temp(self.dataset.valclips,self.dataset.valy), batch_size=self.hparams.batch_size, shuffle=True)
        # return DataLoader(temp(self.dataset.clips,self.dataset.y), batch_size=self.hparams.batch_size, shuffle=True)
    
    def validation_step(self,batch,batch_idx):
        x,y = batch
        y = y.cuda() if self.hparams.full_gpu else y
        y_hat = self.forward(x)
        tensor_size = y_hat.shape[0]
        ones = torch.ones(1,tensor_size).cuda() if self.hparams.full_gpu else torch.ones(1,tensor_size) 
        zeros = torch.zeros(1,tensor_size).cuda() if self.hparams.full_gpu else torch.zeros(1,tensor_size)        
        y_hat_rounded = torch.where(y_hat == torch.max(y_hat),ones,zeros)
        
        accu = torch.equal(y_hat_rounded,y)
        return {'val_accu': accu}
    
    def validation_epoch_end(self, outputs):
        avg_accu = [x['val_accu'] for x in outputs]
        avg_accu = sum(avg_accu)/len(avg_accu)
        self.valResults.append(avg_accu)
        print('\nAccuracy:',avg_accu)
        return {'val_accu': avg_accu}

if __name__ == '__main__':
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = ArgumentParser(add_help=False)

    
    # allow model to overwrite or extend args
    parser = add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    #Check to see if inter or intra or inter+intra
    if(not hyperparams.inter and not hyperparams.intra):
        class noTypeSelected(Exception):
            def __init__(self,s):
                pass
        raise noTypeSelected("Must select at least one: --inter, --intra")
            
    accuList = []
    classifier.dataset.KFoldLength = hyperparams.kfold
    for i in range(classifier.dataset.KFoldLength):
        #Set Seeds
        torch.manual_seed(0)
        np.random.seed(0)
            
        rnModel = SuperRNModel(hyperparams)

        early_stop_callback = EarlyStopping(
            monitor='val_accu',
            min_delta=0.01,
            patience=hyperparams.patience,
            verbose=False,
            mode='max'
        )

        trainer = Trainer(max_nb_epochs=hyperparams.epochs, early_stop_callback=False, checkpoint_callback=None)


        trainer.fit(rnModel)
        if(hyperparams.kfold > 1):            
            result = max(rnModel.valResults[5:])
            accuList.append(result)
            print('KFold %d/%d: Accuracy = '%(i,classifier.dataset.KFoldLength),result)

    print('Done!')
    
    if(hyperparams.kfold > 1):
        print('Global Accuracy:',sum(accuList)/len(accuList))

        
#python SuperRNModel.py --intra --changeOrder --randomJointOrder 1 --epochs 20 --kfold 10



