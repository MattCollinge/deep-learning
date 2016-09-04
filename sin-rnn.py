# Simple example using recurrent neural network to predict time series values

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
from tflearn.optimizers import SGD
import numpy as np
import math
import matplotlib
import time

matplotlib.use('Agg')
#%matplotlib inline
import matplotlib.pyplot as plt

def shift(data, amount):
    data = data[:-amount]
    pad = amount
    return np.pad(data, pad, mode='constant', constant_values=np.nan)
#http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
# def sin_cos(x):
#     return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)

# X, y = generate_data(sin_cos, np.linspace(0, 100, 10000), TIMESTEPS, seperate=False)
def gen_data(func, x, steps_of_history, steps_in_future):

    print('pre func x:', np.shape(x))
    x = func(x)
    print('post func x:', np.shape(x))
    seq = []
    next_val = []

    for i in range(0, len(x) - steps_of_history - steps_in_future, 1):
        seq.append(x[i: i + steps_of_history])
        next_val.append(x[i + steps_of_history + steps_in_future - 1])

    print('seq pre reshape:', np.shape(seq))
    seq = np.reshape(seq, [-1, steps_of_history, 1])
    next_val = np.reshape(next_val, [-1, 1])
    print('seq post reshape:', np.shape(seq))

    trainX = np.array(seq)
    trainY = np.array(next_val)
    return trainX, trainY


step_radians = 0.00005
steps_of_history = 100
cell_count_l1 = steps_of_history / 2
cell_count_l2 = cell_count_l1
cell_count_l3 = cell_count_l2
steps_in_future = 10
index = 0
run_id = 'sin-rnn-d2-15epoc' + str(time.time())

func = lambda x: x * np.sin(x**2)

x = np.arange(0, 5*math.pi, step_radians)
valid_x = np.arange(5*math.pi, 7*math.pi, step_radians)

trainX, trainY = gen_data(func, x, steps_of_history, steps_in_future)
validX, validY = gen_data(func, valid_x, steps_of_history, steps_in_future)

# Network building
net = tflearn.input_data(shape=[None, steps_of_history, 1])
net = tflearn.lstm(net, cell_count_l1, dropout=(1.0,1.0), weights_init="xavier", return_seq=True)
net = tflearn.lstm(net, cell_count_l2, dropout=(1.0,1.0), weights_init="xavier", return_seq=True)
net = tflearn.lstm(net, cell_count_l3, dropout=(1.0,1.0), weights_init="xavier")

net = tflearn.fully_connected(net, 128, activation='relu')
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 1, activation='linear')

# activation options: linear, tanh, sigmoid, softmax, softplus, softsign, relu,relu6, leaky_relu, prelu, elu- see: http://tflearn.org/activations/
# loss options: mean_square, hinge_loss, roc_auc_score, softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, weak_cross_entropy_2d - see: http://tflearn.org/objectives/
# optimisers: SGD, RMSProp, Adam, Momentum, AdaGrad, Ftrl, AdaDelta,
sgd = SGD(learning_rate=0.5, lr_decay=0.96, decay_step=100)

net = tflearn.regression(net, optimizer=sgd, loss='mean_square')

# Training
model = tflearn.DNN(net, clip_gradients=0.2, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=15, validation_set=(validX, validY), batch_size=256,
            run_id=run_id)

# Testing
test_x = np.arange(7*math.pi, 8*math.pi, step_radians)
testX, _ = gen_data(func, test_x, steps_of_history, 0)
print(np.shape(testX))

# Predict the future values
predictY = model.predict(testX)

print('y:', np.shape(predictY), predictY[0:10])

predictY = shift(np.array(predictY)[:,0], steps_of_history)
# print(predictY)
plotx = func(test_x)
print ('x:', np.shape(plotx))
print('y:', np.shape(predictY), predictY[steps_of_history + steps_in_future:steps_of_history + steps_in_future + 10])

# Plot the results
plt.figure(figsize=(20,4))
plt.suptitle('Prediction')
plt.title('History='+str(steps_of_history)+', Future='+str(steps_in_future) + ', TBd_run=' + run_id)
plt.plot(plotx, 'r-', label='Actual')
plt.plot(predictY, 'g--', label='Predicted')
plt.legend('')
plt.savefig(run_id + '.png')
