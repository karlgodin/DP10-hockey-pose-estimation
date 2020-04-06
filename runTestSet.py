import os, sys
from argparse import ArgumentParser
from classifier.joints import parse_clip

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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger


import numpy as np
import math
import shlex



def accuracy(y_pred, y):
    correct = (y == y_pred).sum().float()
    acc = correct/len(y)
    return acc



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
    parser.add_argument('--resume_from_checkpoint', default=None, type=str)
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
            self.ones = torch.ones(1,2).cuda() if self.hparams.full_gpu else torch.ones(1,2)
            self.zeros = torch.zeros(1,2).cuda() if self.hparams.full_gpu else torch.zeros(1,2)
            
        elif hparams.dataset == "SBU":
            self.criterion = nn.CrossEntropyLoss()
            self.dataset = SBUDataset('classifier/SBUDataset', hparams)
            self.ones = torch.ones(1,8).cuda() if self.hparams.full_gpu else torch.ones(1,8)
            self.zeros = torch.zeros(1,8).cuda() if self.hparams.full_gpu else torch.zeros(1,8)

        #Load dataset and find its size
        numOfFrames = self.dataset.numOfFrames
        numOfJoints = self.dataset.numOfJoints
        
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
        
        if(self.isInter):
            input_data_clip_combinations = x
                       
            tensor_g = self.g_model_inter(input_data_clip_combinations)

            # calculate sum and div
            sum = torch.sum(tensor_g, dim=1)
            size_output_G = tensor_g.shape[1]
            average_output = sum / size_output_G

            tensor_classification = self.f_model(average_output)
            
        if(self.isIntra):
            input_data_clip_combinations_P1 = x[0]
            input_data_clip_combinations_P2 = x[1]
                            
            tensor_g_P1 = self.g_model_intra(input_data_clip_combinations_P1)
            tensor_g_P2 = self.g_model_intra(input_data_clip_combinations_P2)

            # calculate sum and div
            average_output = torch.empty((0))
            if(self.hparams.full_gpu):
                average_output = average_output.cuda()
            
            for tensor_g in [tensor_g_P1,tensor_g_P1]:
                sumTensorG = torch.sum(tensor_g, dim=1)
                size_output_G = tensor_g.shape[1]
                average_output_temp = sumTensorG / size_output_G
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
        y_hat = self.forward(x)

        trainAccuracy = []
        for validated, label in zip(y_hat, y):
            y_hat_rounded = torch.where(validated == torch.max(validated), self.ones, self.zeros).squeeze(0)
            accu = torch.equal(y_hat_rounded, label)
            trainAccuracy.append(accu)

        trainAccuracy = sum(trainAccuracy) / len(trainAccuracy) if len(trainAccuracy) != 0 else 0.0

        if self.hparams.dataset == "PHYT":
            loss = self.criterion(y_hat.squeeze(), y.squeeze())
        elif self.hparams.dataset == "SBU":
            loss = self.criterion(y_hat, torch.max(y, 1)[1])

        tensorboard_logs = {'train_loss': loss, 'train_acc': trainAccuracy}
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
               
        valAccuracy = []
        for validated,label in zip(y_hat,y):
            y_hat_rounded = torch.where(validated == torch.max(validated),self.ones,self.zeros).squeeze(0)
            accu = torch.equal(y_hat_rounded,label)
            valAccuracy.append(accu)

        valAccuracy = sum(valAccuracy)/len(valAccuracy) if len(valAccuracy) != 0 else 0.0

        if(self.hparams.dataset == 'PHYT'):
          loss = self.criterion(y_hat.squeeze(), y.squeeze())
        elif(self.hparams.dataset == 'SBU'):
          _,labels = torch.max(y.squeeze(),1)
          loss = self.criterion(y_hat.squeeze(),labels)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'val_accu': valAccuracy, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_accu = [x['val_accu'] for x in outputs]
        avg_accu = sum(avg_accu)/len(avg_accu)

        avg_loss = [x['val_loss'] for x in outputs]
        avg_loss = sum(avg_loss) / len(avg_loss)

        self.valResults.append(avg_accu)
        print('\nAccuracy:',avg_accu)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_accu}

        return {'val_accu': avg_accu, 'val_loss': avg_loss, 'log': tensorboard_logs}

if __name__ == '__main__':
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = ArgumentParser(add_help=False)
    version = None
    
    # allow model to overwrite or extend args
    parser = add_model_specific_args(parent_parser, root_dir)

    folder = sys.argv[1]
    versionNum = folder.split('/')[-1]
    checkpoint = [f for f in os.listdir(folder) if '_ckpt_epoch' in f ][0]

    with open(folder +'/meta_tags.csv') as f:
      next(f)
      cmd = ''
      for line in f:
        key,value = line.strip().split(',')
        if(value == 'True'):
          cmd += '--{} '.format(key)
        elif(value == 'False'):
          pass
        elif(value == ''):
          pass
        else:
          cmd += '--{} {} '.format(key,value)

    cmd += '--resume_from_checkpoint "{}"'.format(folder+'/%s'%checkpoint)
    cmd = shlex.split(cmd)
    hyperparams = parser.parse_args(cmd)

    classifier.dataset.KFoldLength = hyperparams.kfold
    resume_from_checkpoint = hyperparams.resume_from_checkpoint

    #Check to see if inter or intra or inter+intra
    if(not hyperparams.inter and not hyperparams.intra):
        class noTypeSelected(Exception):
            def __init__(self,s):
                pass
        raise noTypeSelected("Must select at least one: --inter, --intra")
            

    pretrainedModel = SuperRNModel.load_from_checkpoint(hyperparams.resume_from_checkpoint)

    predList = []
    for x,y in zip(pretrainedModel.dataset.testclips,pretrainedModel.dataset.testy):
      y_hat = pretrainedModel.forward(x.unsqueeze(0))    
      _,predIdx = y_hat.max(1)
      _,truthIdx = y.max(0)
      predList.append(predIdx.cpu().numpy().tolist()[0] == truthIdx.cpu().numpy().tolist())

print('Test Accuracy for %s:'%versionNum, sum(predList)/len(predList))
with open(folder+'/testAccuracy.txt','w') as f:
  f.write(str(sum(predList)/len(predList)))

#python SuperRNModel.py --intra --changeOrder --randomJointOrder 1 --epochs 20 --kfold 10
