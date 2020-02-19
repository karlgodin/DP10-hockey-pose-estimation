import os, sys
from argparse import ArgumentParser
from classifier.joints import parse_clip
from classifier.GTheta import get_combinations
from classifier.GTheta import GTheta
from classifier.FPhi import FPhi
import torch.tensor

# Pytorch Lightning
import pytorch_lightning as pl

import numpy as np

class SuperRNModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SuperRNModel, self).__init__()
        self.g_model = GTheta(hparams)
        self.f_model = FPhi(hparams)


    def forward(self, perp: np.ndarray, victim: np.ndarray):
        print("Begin: forwarding of the SuperRNModel")

        print("starting calculations for G function")
        # numpy matrix of all combination of inter joints
        inter_combinations_joints = get_combinations(perp, victim)

        # change those values depending on the number of combination and the number of outputs from layer of G
        accumulation = torch.zeros(250, dtype=torch.float)
        for x in range(len(inter_combinations_joints)):

            input_tensor = torch.FloatTensor(inter_combinations_joints[x])
            tensor_g = self.g_model(input_tensor)
            accumulation = accumulation.add(tensor_g)

        # do average
        average = torch.div(accumulation, len(inter_combinations_joints))

        print("starting forward for F function")
        tensor_classification = self.f_model(average)
        return tensor_classification


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
    perp, victim = parse_clip()

    # allow model to overwrite or extend args
    parser = GTheta.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()


    rnModel = SuperRNModel(hyperparams)
    rnModel.train()
    results = rnModel(perp, victim)
    print(results)
    # train model
    #main(hyperparams, None)


