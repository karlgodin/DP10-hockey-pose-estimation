import itertools as it
import sys
import random
import os

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
optim = ['Adam']

def getGridSearchFromParameterList():
        
    #optim = Adam
    combinationsAdam = it.product(*(set,patience,kfold,epochs,optim,lr,batch_size,randomJointOrder))
    #optim = SGD
    #combinationsSGD = it.product(*(set,patience,kfold,epochs,['SGD'],lr,batch_size,randomJointOrder,momentum,nesterov))
    
    #concat lists
    combinations = list(combinationsAdam)
    return list(combinations)
    
def getGridSearchFromBestRuns(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    
    #Columns of attributes
    wantedColumns = [col for col in df.columns if 'Column' in col]
    
    #Create list of runs
    runs = []
    for idx in df.index:
        params = {}
        for col in wantedColumns:
            txt = df.loc[idx][col]
            key,value = txt.replace("'","").replace(" ",'').split(':')
            params[key] = value
        
        combo = []
        combo.append(str(set[0]))
        combo.append(params['patience']) if 'patience' in params else combo.append(str(patience[0]))
        combo.append(params['kfold']) if 'kfold' in params else combo.append(str(kfold[0]))
        combo.append(params['epochs']) if 'epochs' in params else combo.append(str(epochs[0]))
        combo.append(params['optim']) if 'optim' in params else combo.append(str(optim[0]))
        combo.append(params['lr']) if 'lr' in params else combo.append(str(lr[0]))
        combo.append(params['batch_size']) if 'batch_size' in params else combo.append(str(batch_size[0]))
        combo.append(params['randomJointOrder']) if 'randomJointOrder' in params else combo.append(str(randomJointOrder[0]))
            
        runs.append(combo)
    
    
    #Take only ten best
    k=10
    runs = runs[:k]
    
    #For each run, multiply LR by 10 and 0.1. In total, runs should have len 3xk 
    for i in range(k):
        tempCombo = runs[i].copy()
        LR = float(tempCombo[5])
        
        #LR x 10
        tempCombo[5] = str(LR * 10)
        runs.append(tempCombo.copy())
        
        #LR / 10
        tempCombo[5] = str(LR * 0.01)
        runs.append(tempCombo.copy())
    
    return runs
    

if __name__ == '__main__':
    RERUN = True
    combHyperparams = ''
    cmd_template = "python SuperRNModel.py "
    # knowing the order of the command, the forth argument (not counting python) will always be the optim
    #hyperparamsCombinationList = getGridSearchFromParameterList()
    hyperparamsCombinationList = getGridSearchFromBestRuns('./SortedRuns/run_summary.csv')

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
    #outCommandList = outCommandList[:int(len(outCommandList) * 0.3)]
    
    #Remove commands already made
    members = ['Marine3','Shawn3','Karl3']
    if(not RERUN):
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
            for cmd in commands:
                f.write(cmd[0]) #inter
                f.write('\n')
                f.write(cmd[1]) #intra
                f.write('\n')
        