__author__ = 'PC-LiNing'

import numpy as np

from sklearn.metrics import recall_score,accuracy_score,f1_score
y_true = [0, 3, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print('############')
print(recall_score(y_true, y_pred, average='macro'))
print(accuracy_score(y_true, y_pred))
print(f1_score(y_true, y_pred, average='macro'))
print('############')
print(recall_score(y_true, y_pred, average='micro'))
print(accuracy_score(y_true, y_pred))
print(f1_score(y_true, y_pred, average='micro'))
print('############')
print(recall_score(y_true, y_pred, average='weighted'))
print(accuracy_score(y_true, y_pred))
print(f1_score(y_true, y_pred, average='weighted'))
print('############')
