import random
import json
import pathlib

idxes = list(range(83))
random.seed(0)

for _ in range(10):
    random.shuffle(idxes)

testSets = idxes[:len(idxes)//10]
idxes = idxes[len(idxes)//10:]
out = {'test':testSets,'train':idxes}

with open('./classifier/sets.json','w') as f:
    json.dump(out,f,indent=4)
