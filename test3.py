__author__ = 'PC-LiNing'

import numpy as np

x = np.array([1,2])
print(x.shape)
x = np.expand_dims(x, axis=-1)

print(x.shape)