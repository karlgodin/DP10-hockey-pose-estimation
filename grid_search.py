import itertools as it
import sys
import random
import os

def getGridSearch():
    set = ['inter']
    patience = [10, 25]
    kfold = [5, 10]
    epochs = [300]
    lr = [0.0001, 0.00001, 0.000001]
    batch_size = [1,2,4, 8, 16]
    randomJointOrder = [10]
    momentum = [0.1]
    nesterov = [True, False]

    # always true for now
    changeOrder = ['true', 'false']
    
    #optim = Adam
    combinationsAdam = it.product(*(set,patience,kfold,epochs,['Adam'],lr,batch_size,randomJointOrder))
    #optim = SGD
    #combinationsSGD = it.product(*(set,patience,kfold,epochs,['SGD'],lr,batch_size,randomJointOrder,momentum,nesterov))
    
    #concat lists
    combinations = list(combinationsAdam)
    return list(combinations)

if __name__ == '__main__':
    combHyperparams = ''
    cmd_template = "python SuperRNModel.py "
    # knowing the order of the command, the forth argument (not counting python) will always be the optim
    hyperparamsCombinationList = getGridSearch()

    # loop over all possible combinations of hyperparameters
    outCommandList = []
    for element in hyperparamsCombinationList:
        if 'Adam' == element[4]:
            combHyperparams = '--patience ' + str(element[1]) + ' --kfold ' + str(element[2]) + ' --epochs ' + str(element[3]) + ' --optim ' + str(element[4]) + ' --lr ' + str(element[5]) + ' --batch_size ' + str(element[6]) + ' --randomJointOrder ' + str(element[7])
        elif 'SGD' in element[4]:
            combHyperparams = '--patience ' + str(element[1]) + ' --kfold ' + str(element[2]) + ' --epochs ' + str(
                element[3]) + ' --optim ' + str(element[4]) + ' --lr ' + str(element[5]) + ' --batch_size ' + str(
                element[6]) + ' --randomJointOrder ' + str(element[7]) + ' --momentum ' + str(element[8])

            if element[9] == True:
                combHyperparams = combHyperparams + ' --nesterov'
        else:
            raise(Exception)
        
        default_cmd = cmd_template + "--full_gpu "
        
        
        cmdInter = default_cmd + "--inter" + " --changeOrder --dataset PHYT " + combHyperparams
        cmdIntra = default_cmd + "--intra" + " --changeOrder --dataset PHYT " + combHyperparams
            
        outCommandList.append((cmdInter,cmdIntra))

    random.seed(0) 
    random.shuffle(outCommandList)
    
    #Take only 30% of whole test
    outCommandList = outCommandList[:int(len(outCommandList))]
    
    #Remove commands already made
    members = ['Marine2','Shawn2']
    commandsAlreadyRan = []
    for file in os.listdir('./GridSearchTODOs'):
        if file in list(map(lambda x: "TODO_%s.txt"%x,members)):
            continue
        with open('./GridSearchTODOs/%s'%file) as f:
            for line in f:
                commandsAlreadyRan.append(line)
    outCommandList = [lines for lines in outCommandList if lines[0] + '\n' not in commandsAlreadyRan]
    
    #Write files for each team members
    outCommandLists = [outCommandList[i::len(members)] for i in range(len(members))]
    
    for member, commands in zip(members,outCommandLists):
        with open('GridSearchTODOs/TODO_%s.txt'%member,'w') as f:
            for cmd in commands[:5]:
                f.write(cmd[0]) #inter
                f.write('\n')
                f.write(cmd[1]) #intra
                f.write('\n')
        