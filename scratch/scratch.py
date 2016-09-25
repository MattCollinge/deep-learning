import scipy as sp
import numpy as np

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

print(logloss([1,1,1,1], [1,1,1,0]))


def to_one_hot(dataY):
    """Convert the vector of labels dataY into one-hot encoding.

    :param dataY: vector of labels
    :return: one-hot encoded labels
    """
    nc = int(1 + np.max(dataY))
    print('nc:', nc)
    onehot = [np.zeros(nc, dtype=np.int8) for _ in dataY]
    for i, j in enumerate(dataY):
        onehot[i][j] = 1
    return np.asarray(onehot)

y = np.load('/home/dev/data/numer.ai/2016-09-08/labels-test-2016-09-08.npy')
# print('One hot:', to_one_hot([0,1,2,4.5,1,2]))

print(y)
print('One hot:', to_one_hot(y))