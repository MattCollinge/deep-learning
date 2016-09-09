from __future__ import division, print_function, absolute_import


#!/usr/bin/python
import sys, getopt
import tflearn, numpy as np
import pandas as pd
import time
from datetime import date
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import DataPreprocessing
from tflearn.optimizers import SGD, Adam

def load_data(data_set_name, data_set_type):
	dataBasePath = '/home/dev/data/numer.ai/'+ data_set_name	+'/'
	return np.load(dataBasePath + data_set_type + '-'+ data_set_name + '.npy')

def gen_run_id(data_set_name):
	return 'numerai-' + data_set_name + '-'+ date.today().isoformat() + '-' + str(time.time())
	
def build_model():
	weight_init_strat = 'xavier'
	activation_strat = 'relu'

	# Define our network architecture:
	data_prep = DataPreprocessing()
	data_prep.add_featurewise_zero_center() #mean=0.499411645801) #mean=0.499414771557)
	data_prep.add_featurewise_stdnorm() #std=0.291343533186) #std=0.291349126404)
	# X = tflearn.data_utils.featurewise_zero_center(X)
	# X = tflearn.data_utils.featurewise_std_normalization(X)

	network = input_data(shape=[None, 21, 1], data_preprocessing=data_prep)

	# Step 1: Convolution

	network = conv_1d(network, 128, 16, activation=activation_strat, weights_init=weight_init_strat)
	network = max_pool_1d(network, 2)

	network = conv_1d(network, 256, 11, activation=activation_strat, weights_init=weight_init_strat)

	network = conv_1d(network, 512, 11, activation=activation_strat, weights_init=weight_init_strat)
	network = conv_1d(network, 512, 11, activation=activation_strat, weights_init=weight_init_strat)
	network = conv_1d(network, 512, 11, activation=activation_strat, weights_init=weight_init_strat)

	network = max_pool_1d(network, 2)

	network = conv_1d(network, 384, 6, activation=activation_strat, weights_init=weight_init_strat)
	network = max_pool_1d(network, 2)
	network = conv_1d(network, 64, 3, activation=activation_strat, weights_init=weight_init_strat)
	#network = max_pool_1d(network, 2)
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
	# network = dropout(network, 0.1)
	network = fully_connected(network, 4096, activation=activation_strat, weights_init=weight_init_strat)
	network = dropout(network, 0.2) #0.4 and 0.5 later is best
	#network = fully_connected(network, 4096, activation=activation_strat, weights_init=weight_init_strat)
	#network = dropout(network, 0.7)

	network = fully_connected(network, 512, activation=activation_strat, weights_init=weight_init_strat)
	network = dropout(network, 0.2) #0.3 for both looks promising

	# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
	network = fully_connected(network, 2, activation='softmax', restore=True, weights_init=weight_init_strat)

	sgd = SGD(learning_rate=0.5, lr_decay=0.9, decay_step=100, staircase=False)
	# adam = Adam(learning_rate=1.5, epsilon=0.1,)
	# Tell tflearn how we want to train the network
	network = regression(network, optimizer=sgd,
	                     loss='categorical_crossentropy')

	# Wrap the network in a model object
	model = tflearn.DNN(network, tensorboard_verbose=0)
	return model
	                    # ,max_checkpoints=15,
	                    # checkpoint_path='/home/dev/data-science/next-interval-classifier.checkpoints/next-interval-classifier-50k-grey-closeonly.tfl.ckpt')
def train(model, data_set_name):
	#Hyperparamters
	validationPC = 0.1
	batch_size = 200
	epochs = 160

	run_id = gen_run_id(data_set_name)
	
	X = load_data(data_set_name, 'features')
	Y = load_data(data_set_name, 'labels')

	Y = to_categorical(Y, 2)

	# Train it! We'll do 100 training passes and monitor it as it goes.
	model.fit(X, Y, n_epoch=epochs, shuffle=True, validation_set=validationPC,
	          show_metric=True, batch_size=batch_size,
	        #   snapshot_epoch=True,
	          run_id=run_id)

	# Save model when training is complete to a file
	model.save('numerai-' + data_set_name + '.tfl')
	# print("Network trained and saved as next-interval-classifier-grey-150k-closeonly.tfl!")

def predict(model, data_set_name):
	print ('Predicting on: ', data_set_name)
	from sklearn.metrics import classification_report	
	model.load('numerai-' + data_set_name + '-best.tfl')#, weights_only=True)
	
	X = load_data(data_set_name, 'tournament')
	ids = load_data(data_set_name,'tournament_ids')

	# print(np.shape(X), np.shape(ids))
	# print(ids[0])
	batch_size = 2000
	df = pd.DataFrame(ids)
	predictions = np.zeros(len(df))
	# print(df.values[0:10,:])
	# Need to cycle round predicting each batch and appending it to an array for F1 scoring
	for x in xrange(0,len(X), batch_size):
		predictOutput = np.array(model.predict(X[x:x+batch_size]))
		for p in xrange(0,len(predictOutput)):
			predictions[x+p] = predictOutput[p,1]
		
		print(ids[x:x+1,0], predictions[x:x+1])

	df = pd.DataFrame({'t_id': ids[:,0], 'probability': predictions})
	print(df.head())
	df.to_csv(data_set_name + '-prediction.csv', index=False)
	#print(classification_report(Y, y_pred))


def main(argv):
	data_set_name = ''
	model_mode =  'train'
	try:
	  opts, args = getopt.getopt(argv,"hm:d:",["mode=","dataset="])
	except getopt.GetoptError:
	  print ('numer.ai.py -m <train || predict> -d <dataset_name>')
	  sys.exit(2)
	for opt, arg in opts:
	  if opt == '-h':
		print ('numer.ai.py -m <train || predict> -d <dataset_name>')
		sys.exit()
	  elif opt in ("-m", "--mode"):
	     model_mode = arg
	  elif opt in ("-d", "--dataset"):
	     data_set_name = arg
	model = build_model()
	if model_mode == 'predict':
		print ('Predicting on: ', data_set_name)
		predict(model, data_set_name)
	else:
		model_mode = 'train'
		print ('Training on: ', data_set_name)
		train(model, data_set_name)
   	
   
if __name__ == "__main__":
   main(sys.argv[1:])

# data_set_name = '2016-09-08'
# model_mode = 'train'
# model = build_model(data_set_name)
# print ('Training on: ', data_set_name)
# train(model)

# from sklearn.metrics import classification_report
# import tflearn, numpy as np
# from tflearn.data_utils import shuffle, to_categorical
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.estimator import regression
# from tflearn.data_preprocessing import ImagePreprocessing
# from tflearn.layers.normalization import local_response_normalization

# #Set up some variables for the Evaluation of the Models predicitve power...
# newDatacsv = '/home/dev/data/EURUSD_Candlestick_5_m_BID_01.08.2016-19.08.2016.csv'
# dataUnseenBasePath = '/home/dev/data/prep-5M-EURUSD-Unseen-2016-08-19/'

# model.load(modelPath + 'next-interval-classifier-50k-grey-closeonly.tfl.ckpt-33675', weights_only=True)
# Y = np.load(dataUnseenBasePath + 'labelsRaw.npy')
# X = np.load(dataUnseenBasePath + 'raw.npy')
# Y_eval = to_categorical(Y, 2)

# print np.shape(X), np.shape(Y)
# batchSize = 10
# # Need to cycle round predicting each batch and appending it to an array for F1 scoring


# # Feed new, unseen data to model and see how it performs...
# predictOutput = np.array(model.predict(X))
# #evaluateOutput = model.evaluate(X,Y_eval)
# del model

# print predictOutput
# y_pred = np.around(predictOutput[:,0])
# print y_pred
# print(classification_report(Y, y_pred))

