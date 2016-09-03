from __future__ import division, print_function, absolute_import
#Hyperparamters
validationPC = 0.1


# Import tflearn and some helpers
import tflearn, numpy as np
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

# Load path/class_id image file:
dataBasePath = '/home/dev/data/prep-5M-EURUSD-50k/'
dataset_file = dataBasePath + 'labels.txt'

# Build the preloader array, resize images to 128x128
from tflearn.data_utils import image_preloader
#X, Y = image_preloader(dataset_file, image_shape=(128, 128),   mode='file', categorical_labels=True,   normalize=True)
Y = np.load(dataBasePath + 'labelsRaw.npy')
X = np.load(dataBasePath + 'raw.npy')

Y = to_categorical(Y, 2)

# Make sure the data is normalized
img_prep = ImagePreprocessing()
# Full ~492k data set
#img_prep.add_featurewise_zero_center(mean=0.254854548285)
#img_prep.add_featurewise_stdnorm(std=0.433881521497)

# 1st 50K of dataset
#img_prep.add_featurewise_zero_center(mean=-0.434702)
#img_prep.add_featurewise_stdnorm(std=0.205186)

#Full Greyscale
#img_prep.add_featurewise_zero_center(mean=4.08979363886)
#img_prep.add_featurewise_stdnorm(std=27.1395597341)

#5k Greyscale
#img_prep.add_featurewise_zero_center(mean=-0.482901)
#img_prep.add_featurewise_stdnorm(std=0.108251)

# 1st 50K of dataset Close Only Grey
img_prep.add_featurewise_zero_center(mean=-0.5)
img_prep.add_featurewise_stdnorm(std=0.000345241)

# Calc each time...
#img_prep.add_featurewise_zero_center()
#img_prep.add_featurewise_stdnorm()

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 128, 128, 1],data_preprocessing=img_prep)

# Step 1: Convolution
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2)
# # Step 4: Convolution yet again
#network = conv_2d(network, 512, 1, activation='relu')
#network = max_pool_2d(network, 2)
# network = conv_2d(network, 384, 6, activation='relu')
# network = conv_2d(network, 384, 3, activation='relu')
#network = max_pool_2d(network, 2)
#network = conv_2d(network, 256, 1, activation='relu')

network = fully_connected(network, 1024, activation='relu')
#network = dropout(network, 0.8)
network = fully_connected(network, 1024, activation='relu')
# network = fully_connected(network, 256, activation='relu')
# network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax', restore=True)

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.000001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, max_checkpoints=15,
                    checkpoint_path='/home/dev/data-science/next-interval-classifier.checkpoints/next-interval-classifier-50k-grey-closeonly.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=15, shuffle=True, validation_set=validationPC,
          show_metric=True, batch_size=20,
          snapshot_epoch=True,
          run_id='next-interval-50k-grey-closeonly-03')

# Save model when training is complete to a file
model.save("next-interval-classifier-grey-50k-closeonly.tfl")
print("Network trained and saved as next-interval-classifier-grey-150k-closeonly.tfl!")
