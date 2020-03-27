import subprocess
import sys
import os

if(len(sys.argv) != 2):
    print('Enter your name as argument.')
    exit()
MYNAME = sys.argv[1].capitalize()
DONEPath = '../drive/My Drive/DesignProject/GridSearchDONEs/DONE%s.txt'%MYNAME

cmds = []
with open('GridSearchTODOs/TODO_%s.txt'%MYNAME,'r') as f:
    for cmd in f:
        cmds.append(cmd)

if(not os.path.exists('../drive/My Drive/DesignProject/')):
  os.makedirs('../drive/My Drive/DesignProject/')

if(os.path.exists(DONEPath)):
    with open(DONEPath) as f:
        done = int(f.read())
        done += 1
else:
    done = 0

for cmd in cmds[done:10]:
    print(cmd)
    subprocess.call(cmd,shell=True)
    
    if(not os.path.exists('../drive/My Drive/DesignProject/GridSearchDONEs')):
        os.mkdir('../drive/My Drive/DesignProject/GridSearchDONEs')
    
    done+=1

    with open(DONEPath,'w') as f:
        f.write(str(done))
    