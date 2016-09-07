from __future__ import division, print_function, absolute_import
#Hyperparamters
validationPC = 0.1


# Import tflearn and some helpers
import tflearn, numpy as np
import time
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import DataPreprocessing
from tflearn.optimizers import SGD, Adam
import pickle

# Load path/class_id image file:
dataBasePath = '/home/dev/data/numer.ai/'
run_id = 'numerai-cnn-2016-09-04-' + str(time.time())
weight_init_strat = 'xavier'
activation_strat = 'relu'
batch_size = 2000
epochs = 50

X = np.load(dataBasePath + 'features-2016-09-04.npy')
Y = np.load(dataBasePath + 'labels-2016-09-04.npy')

Y = to_categorical(Y, 2)

# Define our network architecture:
data_prep = DataPreprocessing()
data_prep.add_featurewise_zero_center(mean=0.499414771557)
data_prep.add_featurewise_stdnorm(std=0.291349126404)
# X = tflearn.data_utils.featurewise_zero_center(X)
# X = tflearn.data_utils.featurewise_std_normalization(X)

network = input_data(shape=[None, 21, 1], data_preprocessing=data_prep)

# Step 1: Convolution

network = conv_1d(network, 128, 16, activation=activation_strat, weights_init=weight_init_strat)
network = max_pool_1d(network, 2)
network = conv_1d(network, 256, 11, activation=activation_strat, weights_init=weight_init_strat)
network = conv_1d(network, 512, 9, activation=activation_strat, weights_init=weight_init_strat)

network = conv_1d(network, 512, 9, activation=activation_strat, weights_init=weight_init_strat)
#network = conv_1d(network, 512, 9, activation=activation_strat, weights_init=weight_init_strat)

network = conv_1d(network, 384, 6, activation=activation_strat, weights_init=weight_init_strat)
network = max_pool_1d(network, 2)
network = conv_1d(network, 64, 3, activation=activation_strat, weights_init=weight_init_strat)

#fully connected layers
#network = fully_connected(network, 762, activation=activation_strat, weights_init=weight_init_strat)
#network = dropout(network, 0.5)
#network = fully_connected(network, 1024, activation=activation_strat, weights_init=weight_init_strat)
#network = dropout(network, 0.5)
#network = fully_connected(network, 2048, activation=activation_strat, weights_init=weight_init_strat)
#network = dropout(network, 0.5)
#network = fully_connected(network, 4096, activation=activation_strat, weights_init=weight_init_strat)
#network = dropout(network, 0.5)
#network = fully_connected(network, 8192, activation=activation_strat, weights_init=weight_init_strat)
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation=activation_strat, weights_init=weight_init_strat)
network = dropout(network, 0.2) #0.4 and 0.5 later is best
#network = fully_connected(network, 4096, activation=activation_strat, weights_init=weight_init_strat)
#network = dropout(network, 0.7)

network = fully_connected(network, 512, activation=activation_strat, weights_init=weight_init_strat)
network = dropout(network, 0.2) #0.3 for both looks promising

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax', restore=True, weights_init=weight_init_strat)

sgd = SGD(learning_rate=0.7, lr_decay=0.96, decay_step=100)
# adam = Adam(learning_rate=1.5, epsilon=0.1,)
# Tell tflearn how we want to train the network
network = regression(network, optimizer=sgd,
                     loss='categorical_crossentropy')

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0)
                    # ,max_checkpoints=15,
                    # checkpoint_path='/home/dev/data-science/next-interval-classifier.checkpoints/next-interval-classifier-50k-grey-closeonly.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=epochs, shuffle=True, validation_set=validationPC,
          show_metric=True, batch_size=batch_size,
        #   snapshot_epoch=True,
          run_id=run_id)

# Save model when training is complete to a file
# model.save("next-interval-classifier-grey-50k-closeonly.tfl")
# print("Network trained and saved as next-interval-classifier-grey-150k-closeonly.tfl!")
