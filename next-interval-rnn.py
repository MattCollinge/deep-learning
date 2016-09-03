from __future__ import division, print_function, absolute_import

#Hyperparamters
validationPC = 0.1

import tflearn, numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb
from sklearn.preprocessing import MinMaxScaler
# Load path/class_id image file:
dataBasePath = '/home/dev/data/prep-5M-EURUSD-150k/'
dataset_file = dataBasePath + 'labels.txt'

# Build the preloader array, resize images to 128x128
from tflearn.data_utils import image_preloader
#X, Y = image_preloader(dataset_file, image_shape=(128, 128),   mode='file', categorical_labels=True,   normalize=True)

fullTimeSeries = np.load(dataBasePath + 'rawCloseOnly.npy')
# print ('fullTimeSeries shape: ',  np.shape(fullTimeSeries))
# print ('fullTimeSeries example: ',  fullTimeSeries[123])
fullTimeSeries = fullTimeSeries.reshape(-1, 1)
# print ('fullTimeSeries shape: ',  np.shape(fullTimeSeries))
# print ('fullTimeSeries example: ',  fullTimeSeries[123])

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(fullTimeSeries)

def createDataset(rawTSdata, lookback=1):
    dataX, dataY = [], []
    for i in range(len(rawTSdata)-lookback-1):
        a = rawTSdata[i:(i+lookback),0]
        dataX.append(a)
        nexta = rawTSdata[i + lookback, 0]
        y = 0
        if nexta > a[lookback-1]:
            y = 1
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

lookback = 1000

X, Y = createDataset(dataset, lookback)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
# print ('X shape: ',  np.shape(X))
# print ('Y shape: ', np.shape(Y))
#
# print ('X example: ',  X[123])
# print ('Y example: ', Y[123])
# print ('nDim: ', type(X) not in [list, np.array])
Y = to_categorical(Y, nb_classes=2)

# Data preprocessing
# Sequence padding
# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
# trainY = to_categorical(trainY, nb_classes=2)
# testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 1, lookback])
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 4096, dropout=0.9, return_seq=True)#dropout=(0.9, 0.9), forget_bias=0.9, return_seq=True)
net = tflearn.lstm(net, 4096, dropout=0.9, return_seq=True)
net = tflearn.lstm(net, 4096)#, dropout=(0.9, 0.9), forget_bias=0.9)
net = tflearn.fully_connected(net, 512, activation='relu')
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
# Wrap the network in a model object
model = tflearn.DNN(net, tensorboard_verbose=0, max_checkpoints=15,
                    checkpoint_path='/home/dev/data-science/next-interval-rnn.checkpoints/next-interval-rnn-50k.tfl.ckpt')

model.fit(X, Y, n_epoch=15, shuffle=True, validation_set=validationPC,
          show_metric=True, batch_size=500,
          snapshot_epoch=True,
          run_id='next-interval-rnn-150k-03')

# Save model when training is complete to a file
model.save("next-interval-rnn-150k.tfl")
print("Network trained and saved as next-interval-rnn-150k.tfl !")
