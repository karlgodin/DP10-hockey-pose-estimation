import itertools as it
import subprocess
import sys

def getGridSearch(optim):
    set = ['inter', 'intra']
    patience = [10, 25, 50]
    kfold = [5, 10, 15]
    epochs = [100, 150, 200, 250, 300]
    optim = [optim]
    lr = [0.001, 0.0001, 0.00001, 0.000001]
    batch_size = [1, 2, 4, 8, 16]
    randomJointOrder = [1, 5, 10]

    # always true for now
    changeOrder = ['true', 'false']

    my_dict = {'set': set, 'patience': patience, 'kfold': kfold, 'epochs': epochs, 'optim': optim, 'lr': lr, 'batch_size': batch_size, 'randomJointOrder': randomJointOrder}
    combinationsAdams = it.product(*(my_dict[Name] for Name in my_dict))
    return list(combinationsAdams)

if __name__ == '__main__':
    argumentsPassed = sys.argv
    default_cmd = "python SuperRNModel.py "
    hyperparamsCombinationList = getGridSearch('Adam')
    count = 0

    # loop over all possible combinations of hyperparameters
    for element in hyperparamsCombinationList:

        combHyperparams = '--patience ' + str(element[1]) + ' --kfold ' + str(element[2]) + ' --epochs ' + str(element[3]) + ' --optim ' + str(element[4]) + ' --lr ' + str(element[5]) + ' --batch_size ' + str(element[6]) + ' --randomJointOrder ' + str(element[7])
        #subprocess.check_call(["python SuperRNModel.py --intra --changeOrder --randomJointOrder 1 --epochs 20 --kfold 10 --dataset SBU --batch_size 32"])

        if '--full_gpu' in argumentsPassed:
            default_cmd = default_cmd + "--full_gpu "
        if element[0] == 'intra':
            default_cmd = default_cmd + "--intra --changeOrder --dataset PHYT "
        elif element[0] == 'inter':
            default_cmd = default_cmd + "--inter --changeOrder --dataset PHYT "
        command = default_cmd + combHyperparams
        subprocess.call(command, shell=True)

        count = count + 1
        if count == 2:
            break

# second combinations for SGD
# optim = ['SGD']
# momemtum = []
# nesterov = []
#
# my_dict = {'patience': patience, 'kfold': kfold, 'epochs': epochs, 'optim': optim, 'lr': lr, 'batch_size': batch_size, 'momemtum': momemtum, 'nesterov': nesterov}
# combinationsSGD = it.product(*(my_dict[Name] for Name in my_dict))
#
# TODO: add the two together to run over all (Adam, SGD)
# combinations = combinationsAdams.append(combinationsSGD)