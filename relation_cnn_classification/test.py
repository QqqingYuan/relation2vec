__author__ = 'PC-LiNing'

import numpy as np

x_train = np.arange(10).reshape(5,2)
y_train = np.arange(20).reshape(5,4)
data=list(zip(x_train,y_train))
print(len(data))
print(data[0])
for batch in data:
    x_batch, y_batch = batch
    print(x_batch)
    print(y_batch)