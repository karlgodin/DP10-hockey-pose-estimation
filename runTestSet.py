from SuperRNModel import *
import shlex
import json


if __name__ == '__main__':
    torch.manual_seed(260743167)
    np.random.seed(260743167)
    
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = ArgumentParser(add_help=False)
    version = None
    
    # allow model to overwrite or extend args
    parser = add_model_specific_args(parent_parser, root_dir)

    folder = sys.argv[1]
    versionNum = folder.split('/')[-1]

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

    cmd = shlex.split(cmd)
    hyperparams = parser.parse_args(cmd)
    classifier.dataset.KFoldLength = hyperparams.kfold

    #Create Model
    model = SuperRNModel(hyperparams)

    #Load Weights
    model.load_state_dict(torch.load(folder + '/model.save'))

    predList = []
    for x,y in zip(model.dataset.testclips,model.dataset.testy):
      x = list(map(lambda t: t.unsqueeze(0),x))
      y_hat = model.forward(x)    
      _,predIdx = y_hat.max(1)
      _,truthIdx = y.max(0)
      predList.append([predIdx.cpu().numpy().tolist()[0],truthIdx.cpu().numpy().tolist()])

    accuList = list(map(lambda x: x[0] == x[1],predList))
    print('Test Accuracy for %s:'%versionNum, sum(accuList)/len(accuList))
    with open(folder+'/testAccuracy.txt','w') as f:
      f.write(str(sum(accuList)/len(accuList)))
    
    with open(folder+'/testResults.json','w') as f:
      json.dump(predList,f)


