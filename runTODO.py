import subprocess
import sys
import os

if(len(sys.argv) != 2):
    print('Enter your name as argument.')
    exit()
MYNAME = sys.argv[1].capitalize()
DONEPath = 'GridSearchDONEs/DONE%s.txt'%MYNAME

cmds = []
with open('GridSearchTODOs/TODO_%s.txt'%MYNAME,'r') as f:
    for cmd in f:
        cmds.append(cmd)

if(os.path.exists(DONEPath)):
    with open(DONEPath) as f:
        done = int(f.read())
else:
    done = 0

for cmd in cmds[done:]:
    print(cmd)
    subprocess.call(cmd,shell=True)
    
    with open(DONEPath,'w') as f:
        f.write(str(done))
    done+=1
