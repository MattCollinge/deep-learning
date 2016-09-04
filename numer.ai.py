from __future__ import division, print_function, absolute_import
#Hyperparamters
validationPC = 0.1


# Import tflearn and some helpers
import tflearn, numpy as np
import time
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

# Load path/class_id image file:
dataBasePath = '/home/dev/data/numer.ai/'
run_id = 'numerai-2016-09-04-' + str(time.time())

X = np.load(dataBasePath + 'features-2016-09-04.npy')
Y = np.load(dataBasePath + 'labels-2016-09-04.npy')

Y = to_categorical(Y, 2)

# Define our network architecture:

network = input_data(shape=[None, 21, 1])

# Step 1: Convolution
# network = conv_2d(network, 64, 1, activation='relu')
# network = max_pool_2d(network, 2)
# # # Step 4: Convolution yet again
# network = conv_2d(network, 512, 1, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 384, 6, activation='relu')
# network = conv_2d(network, 384, 3, activation='relu')
#network = max_pool_2d(network, 2)
#network = conv_2d(network, 256, 1, activation='relu')

network = fully_connected(network, 512, activation='relu')
# network = dropout(network, 0.8)
network = fully_connected(network, 1024, activation='elu')
# network = dropout(network, 0.8)
network = fully_connected(network, 2048, activation='elu')
#network = dropout(network, 0.8)
network = fully_connected(network, 1024, activation='relu')
network = fully_connected(network, 512, activation='relu')
# network = fully_connected(network, 256, activation='relu')
# network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax', restore=True)

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0)
                    # ,max_checkpoints=15,
                    # checkpoint_path='/home/dev/data-science/next-interval-classifier.checkpoints/next-interval-classifier-50k-grey-closeonly.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=15, shuffle=True, validation_set=validationPC,
          show_metric=True, batch_size=2000,
          snapshot_epoch=True,
          run_id=run_id)

# Save model when training is complete to a file
# model.save("next-interval-classifier-grey-50k-closeonly.tfl")
# print("Network trained and saved as next-interval-classifier-grey-150k-closeonly.tfl!")
