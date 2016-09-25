# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tflearn
import tensorflow

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

def create_and_train_full_autoenc(foundation, hidden_count, X, Y, testX, testY, depth, session=None):
    # Building the encoder
    # with tensorflow.Graph().as_default():
        encoder = tflearn.fully_connected(foundation, 256)
        encoder = tflearn.fully_connected(encoder, hidden_count)

        # Building the decoder
        decoder = tflearn.fully_connected(encoder, 256)
        decoder = tflearn.fully_connected(decoder, 784)


        # Regression, with mean square error
        net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                                 loss='mean_square', metric=None)

        # Training the auto encoder
        model = tflearn.DNN(net, tensorboard_verbose=0, session=session)
        model.fit(X, X, n_epoch=10, validation_set=0.1,
                  run_id="auto_encoder_"+ str(depth), batch_size=256)

        # Encoding X[0] for test
        print("\nTest encoding of X[0]:")

        # New model, re-using the same session, for weights sharing
        encoding_model = tflearn.DNN(encoder, session=model.session)
        print(encoding_model.predict([X[0]]))


        # Testing the image reconstruction on new data (test set)
        print("\nVisualizing results after being encoded and decoded:")
        testX = tflearn.data_utils.shuffle(testX)[0]
        # Applying encode and decode over test set
        encode_decode = model.predict(testX)
        encoded = encoding_model.predict(X)

        # encode_decode: (10000, 64)
        print('encode_decode:', np.shape(encode_decode))
        print('encoded:', np.shape(encoded))
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(testX[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        # f.show()
        plt.draw()
        # plt.waitforbuttonpress()
        plt.savefig('autoenc' + str(depth) + '.png')

        #do it all again:
        depth = depth + 1
        encoder = tflearn.input_data(shape=[None, hidden_count])
        encoder = tflearn.fully_connected(encoder, 256)
        encoder = tflearn.fully_connected(encoder, hidden_count)

        # Building the decoder
        decoder = tflearn.fully_connected(encoder, 256)
        decoder = tflearn.fully_connected(decoder, 784)

        # Regression, with mean square error
        net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                                 loss='mean_square', metric=None)

        # Training the auto encoder
        model = tflearn.DNN(net, tensorboard_verbose=0)
        model.fit(encoded, encoded, n_epoch=10, validation_set=0.1,
                  run_id="auto_encoder_"+ str(depth), batch_size=256)

        # Encoding X[0] for test
        print("\nTest encoding of Xencoded[0]:")

        # New model, re-using the same session, for weights sharing
        encoding_model = tflearn.DNN(encoder, session=model.session)
        print(encoding_model.predict([encoded[0]]))


        # Testing the image reconstruction on new data (test set)
        print("\nVisualizing results after being encoded and decoded:")
        testX = tflearn.data_utils.shuffle(testX)[0]
        # Applying encode and decode over test set
        encode_decode = model.predict(encoded)
        encoded = encoding_model.predict(encoded)
        # encode_decode: (10000, 64)
        print('encode_decode:', np.shape(encode_decode))
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(testX[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        # f.show()
        plt.draw()
        # plt.waitforbuttonpress()
        plt.savefig('autoenc' + str(depth) + '.png')


def create_and_train_autoenc(foundation, hidden_count, X, Y, testX, testY, depth, session=None):
    # Building the encoder
    # with tensorflow.Graph().as_default():
        encoder = tflearn.fully_connected(foundation, 256)
        encoder = tflearn.fully_connected(encoder, hidden_count)

        # Building the decoder
        decoder = tflearn.fully_connected(encoder, 256)
        decoder = tflearn.fully_connected(decoder, 784)


        # Regression, with mean square error
        net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                                 loss='mean_square', metric=None)

        # Training the auto encoder
        model = tflearn.DNN(net, tensorboard_verbose=0, session=session)
        model.fit(X, X, n_epoch=10, validation_set=(testX, testX),
                  run_id="auto_encoder_"+ str(depth), batch_size=256)

        # Encoding X[0] for test
        print("\nTest encoding of X[0]:")

        # New model, re-using the same session, for weights sharing
        encoding_model = tflearn.DNN(encoder, session=model.session)
        print(encoding_model.predict([X[0]]))


        # Testing the image reconstruction on new data (test set)
        print("\nVisualizing results after being encoded and decoded:")
        testX = tflearn.data_utils.shuffle(testX)[0]
        # Applying encode and decode over test set
        encode_decode = model.predict(testX)
        encoded = encoding_model.predict(X)
        # encode_decode: (10000, 64)
        print('encode_decode:', np.shape(encode_decode))
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(testX[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        # f.show()
        plt.draw()
        # plt.waitforbuttonpress()
        plt.savefig('autoenc' + str(depth) + '.png')
        return encoded, encoder, model.session

def gen_filename(depth):
    return '/home/dev/data/numer.ai/autoenc-model-' + str(depth) + '.tfl'

def make_new_autoencoder_graph(foundation, hidden_count):
    encoder = tflearn.fully_connected(foundation, 256, restore=True)
    encoder = tflearn.fully_connected(encoder, hidden_count, restore=True)

    # Building the decoder
    decoder = tflearn.fully_connected(encoder, 256,  restore=False)
    decoder = tflearn.fully_connected(decoder, 784, restore=False)
    return encoder, decoder

def run_pretrain(encoder,decoder,X, testX, depth):
    # Regression, with mean square error
    net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                             loss='mean_square', metric=None)

    # Training the auto encoder
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, X, n_epoch=10, validation_set=(testX, testX),
              run_id="auto_encoder_"+ str(depth), batch_size=256)

    model.save(gen_filename(depth))

    encode_decode = model.predict(testX)
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        a[0][i].imshow(np.reshape(testX[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # f.show()
    plt.draw()
    # plt.waitforbuttonpress()
    plt.savefig('autoenc' + str(depth) + '.png')
    return model

def predict_input(encoder, X, pretrain_session, depth):
    # New model, re-using the same session, for weights sharing
    print("\nTest encoding of X[0]:")
    encoding_model = tflearn.DNN(encoder, session=pretrain_session)
    print(encoding_model.predict([X[0]]))

    encoded = encoding_model.predict(X)
    return encoded

def load_foundation(foundation_model, depth):
    return foundation_model.load(gen_filename(depth))

# w = dnn.get_weights(denselayer.W) # get a dense layer weights
# w = dnn.get_weights(convlayer.b)

depth = 1
input = tflearn.input_data(shape=[None, 784])
encoder, decoder = make_new_autoencoder_graph(input, 128)
model = run_pretrain(encoder, decoder, X, testX, depth)
encoded = predict_input(encoder, X, model.session, depth)

depth = 2
input = load_foundation(model, depth-1)
encoder, decoder = make_new_autoencoder_graph(input, 128)
model = run_pretrain(encoder, decoder, encoded, testX, depth)
encoded = predict_input(encoder, encoded, model.session, depth)

print('Done!')
# encoder = tflearn.input_data(shape=[None, 784])
#
# print('oldX:', np.shape(X))
# print('Y:', np.shape(Y))
# print('testY:', np.shape(testX))
# print('testY:', np.shape(testY))
# newX, encoder, session = create_and_train_full_autoenc(encoder, 64, X, Y, testX, testY, 1)
# newX, encoder, session = create_and_train_autoenc(encoder, 64, X, Y, testX, testY, 1)
#
# print('newX:', np.shape(newX))
# print('Y:', np.shape(Y))
# print('testY:', np.shape(testX))
# print('testY:', np.shape(testY))
# newX, encoder, session = create_and_train_autoenc(encoder, 64, X, Y, testX, testY, 2)
