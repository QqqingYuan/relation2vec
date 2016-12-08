__author__ = 'PC-LiNing'

import numpy as np


def getRandom_vec():
    vec=np.random.rand(10)
    norm=np.sum(vec**2)**0.5
    normalized = vec / norm
    return normalized

shapes =[1,2]

indexs = np.zeros(shape=(2,2),dtype=np.int32)

for idx,label in enumerate(shapes):
    index = [label,(label+1)%3]
    indexs[idx] = np.asarray(index)

print(indexs)
print(indexs.shape)

for idx,row in enumerate(indexs.tolist()):
    if shapes[idx] == row[0]:
        print(row[1])
    else:
        print(row[0])

