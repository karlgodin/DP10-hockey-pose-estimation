import itertools as it
import subprocess
import sys

def getGridSearch(optimizer):
    my_dict = {}
    set = ['inter', 'intra']
    patience = [10, 25, 50]
    kfold = [5, 10, 15]
    epochs = [300]
    optim = [optimizer]
    lr = [0.0001, 0.00001, 0.000001]
    batch_size = [1, 2, 4, 8, 16]
    randomJointOrder = [1, 5, 10]
    momentum = [0.1]
    nesterov = [True, False]

    # always true for now
    changeOrder = ['true', 'false']

    if optimizer == 'Adam':
        my_dict = {'set': set, 'patience': patience, 'kfold': kfold, 'epochs': epochs, 'optim': optim, 'lr': lr, 'batch_size': batch_size, 'randomJointOrder': randomJointOrder}
    elif optimizer == "SGD":
        my_dict = {'set': set, 'patience': patience, 'kfold': kfold, 'epochs': epochs, 'optim': optim, 'lr': lr,
                   'batch_size': batch_size, 'randomJointOrder': randomJointOrder, 'momentum': momentum, 'nesterov': nesterov}
    combinations = it.product(*(my_dict[Name] for Name in my_dict))
    return list(combinations)

if __name__ == '__main__':
    combHyperparams = ''
    argumentsPassed = sys.argv
    default_cmd = "python SuperRNModel.py "
    # knowing the order of the command, the forth argument (not counting python) will always be the optim
    hyperparamsCombinationList = getGridSearch(str(argumentsPassed[2]))

    # break the list of combinations to run fewer at a time
    hyperparamsCombinationList = hyperparamsCombinationList[int(argumentsPassed[3]): int(argumentsPassed[4])]

    # loop over all possible combinations of hyperparameters
    for element in hyperparamsCombinationList:
        if 'Adam' in argumentsPassed:
            combHyperparams = '--patience ' + str(element[1]) + ' --kfold ' + str(element[2]) + ' --epochs ' + str(element[3]) + ' --optim ' + str(element[4]) + ' --lr ' + str(element[5]) + ' --batch_size ' + str(element[6]) + ' --randomJointOrder ' + str(element[7])
        elif 'SGD' in argumentsPassed:
            combHyperparams = '--patience ' + str(element[1]) + ' --kfold ' + str(element[2]) + ' --epochs ' + str(
                element[3]) + ' --optim ' + str(element[4]) + ' --lr ' + str(element[5]) + ' --batch_size ' + str(
                element[6]) + ' --randomJointOrder ' + str(element[7]) + ' --momentum ' + str(element[8])

            if element[9] == True:
                combHyperparams = combHyperparams + ' --nesterov'

        if '--full_gpu' in argumentsPassed:
            default_cmd = default_cmd + "--full_gpu "
        if element[0] == 'intra':
            default_cmd = default_cmd + "--intra --changeOrder --dataset PHYT "
        elif element[0] == 'inter':
            default_cmd = default_cmd + "--inter --changeOrder --dataset PHYT "
        command = default_cmd + combHyperparams
        subprocess.call(command, shell=True)
