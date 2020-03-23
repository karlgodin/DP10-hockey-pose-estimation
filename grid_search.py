import itertools as it
import subprocess

def getGridSearch(optim):
    patience = [10, 25, 50]
    kfold = [5, 10, 15]
    epochs = [100, 150, 200, 250, 300]
    optim = [optim]
    lr = [0.001, 0.0001, 0.00001, 0.000001]
    batch_size = [1, 2, 4, 8, 16]

    changeOrder = ['true', 'false']
    randomJointOrder = []

    my_dict = {'patience': patience, 'kfold': kfold, 'epochs': epochs, 'optim': optim, 'lr': lr, 'batch_size': batch_size}

    allNames = sorted(my_dict)
    combinationsAdams = it.product(*(my_dict[Name] for Name in allNames))

    # print(list(combinationsAdams))
    # print(len(list(combinationsAdams)))

    subprocess.check_output(["echo", "Hello World!"])

    return list(combinationsAdams)

if __name__ == '__main__':
    hyperparamsCombinationList = getGridSearch('Adam')
    count = 0
    for element in hyperparamsCombinationList:

        combHyperparams = '--batch_size ' + str(element[0]) + ' --epoch ' + str(element[1]) + ' --kfold ' + str(element[2]) + ' --lr ' + str(element[3]) + ' --optim ' + str(element[4]) + ' --patience ' + str(element[5])
        #subprocess.check_call(["python SuperRNModel.py --intra --changeOrder --randomJointOrder 1 --epochs 20 --kfold 10 --dataset SBU --batch_size 32"])

        default_cmd = "python SuperRNModel.py --intra --changeOrder --randomJointOrder 1 --dataset PHYT "
        command = default_cmd + combHyperparams
        subprocess.call(command)
        count = count + 1
        if count == 2:
            break
    # python
    # SuperRNModel.py - -intra - -changeOrder - -randomJointOrder
    # 1 - -epochs
    # 20 - -kfold
    # 10 - -dataset
    # SBU - -full_gpu - -batch_size
    # 32

# second combinations for SGD
# optim = ['SGD']
# momemtum = []
# nesterov = []
#
# my_dict = {'patience': patience, 'kfold': kfold, 'epochs': epochs, 'optim': optim, 'lr': lr, 'batch_size': batch_size, 'momemtum': momemtum, 'nesterov': nesterov}
# allNames = sorted(my_dict)
# combinationsSGD = it.product(*(my_dict[Name] for Name in allNames))
# print(len(list(combinationsSGD)))
# print(list(combinationsSGD))
#
# combinations = combinationsAdams.append(combinationsSGD)
# print(len(list(combinations)))