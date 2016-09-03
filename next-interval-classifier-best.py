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
dataBasePath = '/home/dev/data/prep-5M-EURUSD-06/'
dataset_file = dataBasePath + 'labels.txt'

# Build the preloader array, resize images to 128x128
from tflearn.data_utils import image_preloader
#X, Y = image_preloader(dataset_file, image_shape=(128, 128),   mode='file', categorical_labels=True,   normalize=True)
Y = np.load(dataBasePath + 'labelsRaw.npy')
X = np.load(dataBasePath + 'raw.npy')

# Build neural network and train

#validationSize = np.round(len(Y) * validationPC)

# Data loading and preprocessing
# Shuffle the data
#X, Y = shuffle(X, Y)

Y = to_categorical(Y, 2)
# Y_test = to_categorical(Y_test, 10)

# Preprocessing... Calculating mean over all dataset (this may take long)...
# Mean: 0.254854548285 (To avoid repetitive computation, add it to argument 'mean' of `add_featurewise_zero_center`)
# ---------------------------------
# Preprocessing... Calculating std over all dataset (this may take long)...
# STD: 0.433881521497 (To avoid repetitive computation, add it to argument 'std' of `add_featurewise_stdnorm`)
# ----------

# Make sure the data is normalized
img_prep = ImagePreprocessing()
# Full ~492k data set
#img_prep.add_featurewise_zero_center(mean=0.254854548285)
#img_prep.add_featurewise_stdnorm(std=0.433881521497)

# 1st 50K of dataset
#img_prep.add_featurewise_zero_center(mean=0.00762648363661)
#img_prep.add_featurewise_stdnorm(std=0.0716199715416)

#Full Greyscale
#img_prep.add_featurewise_zero_center(mean=4.08979363886)
#img_prep.add_featurewise_stdnorm(std=27.1395597341)

#5k Greyscale
#img_prep.add_featurewise_zero_center(mean=-0.482901)
#img_prep.add_featurewise_stdnorm(std=0.108251)


# Calc each time...
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 1],
                     data_preprocessing=img_prep)

# Step 1: Convolution
network = conv_2d(network, 32, 9, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 256, 5, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 512, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 1024, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax', restore=False)

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=1,
                    checkpoint_path='/home/dev/data-science/next-interval-classifier.checkpoints/next-interval-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=15, shuffle=True, validation_set=validationPC,
          show_metric=True, batch_size=200,
          snapshot_epoch=True,
          run_id='next-interval-classifier-1chan_00')

# Save model when training is complete to a file
model.save("next-interval-classifier.tfl")
print("Network trained and saved as next-interval-classifier.tfl!")
