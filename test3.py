__author__ = 'PC-LiNing'

import numpy as np

test1 = np.array([[1,2,3],[4,5,6],[1,1,1],[2,2,2],[3,3,3]])
test2 = np.array([[9,8,7],[6,5,4]])
test3 = np.concatenate((test1,test2))
print(test3)

print(test1[:2])
print(test1[2:])