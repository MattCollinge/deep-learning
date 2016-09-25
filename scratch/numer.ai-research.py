# from bhtsne import run_bh_tsne
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from datetime import date

matplotlib.use('Agg')
# %matplotlib inline

plt.close('all')

numDims = 2
initialDims = 21
perplexity = 30
theta = .5
alg = 'svd'
dataBasePath = '/home/dev/data/numer.ai/'
data_set_name = '2016-09-08'
batch_size = 5000


def save_data(dataset, data_set_name, data_set_type, dataBasePath):
    fullPath = dataBasePath + data_set_name + '/'
    return np.save(fullPath + data_set_type + '-' +
                   data_set_name + '.npy', dataset)


def load_data(data_set_name, data_set_type, dataBasePath):
    dataBasePath = '/home/dev/data/numer.ai/' + data_set_name + '/'
    return np.load(dataBasePath + data_set_type + '-' + data_set_name + '.npy')


def plot_model(embedding, labels):
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                marker='o', s=1, edgecolor='', c=labels)
    fig.tight_layout()


def plot_differences(embedding, actual, lim=1000):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    for a, b in zip(embedding, actual)[:lim]:
        ax.add_line(Line2D((a[0], b[0]), (a[1], b[1]), linewidth=1))
    ax.autoscale_view()
    plt.show()


def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P


def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    # Initialize some variables
    n = X.shape[0]                     # number of instances
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)

    # Compute pairwise distances
    if verbose > 0:
        print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1)
    # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
    D = sum_X + sum_X[:, None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0:
        print('Computing P-values...')
    for i in range(n):

        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))

        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')

        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, indices] = thisP

    if verbose > 0:
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))

    return P, beta


def compute_joint_probabilities(samples, batch_size=5000, d=2,
                                perplexity=30, tol=1e-5, verbose=0):
    v = d - 1

    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)

    # Precompute joint probabilities for all batches
    if verbose > 0:
        print('Precomputing P-values...')
    batch_count = int(n / batch_size)
    P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):
        curX = samples[start:start + batch_size]                 # select batch
        P[i], beta = x2p(curX, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
        P[i][np.isnan(P[i])] = 0                                 # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T) # / 2                             # make symmetric
        P[i] = P[i] / P[i].sum()                                 # obtain estimation of joint probabilities
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    return P


# P is the joint probabilities for this batch (Keras loss functions call this y_true)
# activations is the low-dimensional output (Keras loss functions call this y_pred)
def tsne(P, activations):
    # d = K.shape(activations)[1]
    d = 2  # TODO: should set this automatically, but the above is very slow for some reason
    n = batch_size  # TODO: should set this automatically
    v = d - 1.
    eps = K.variable(10e-15)  # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)
    sum_act = K.sum(K.square(activations), axis=1)
    Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(activations, K.transpose(activations))
    Q = (sum_act + Q) / v
    Q = K.pow(1 + Q, -(v + 1) / 2)
    Q *= K.variable(1 - np.eye(n))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


X_train = load_data(data_set_name, 'features-train', dataBasePath)
Y_train = load_data(data_set_name, 'labels-train', dataBasePath)

X_test = load_data(data_set_name, 'features-test', dataBasePath)
Y_test = load_data(data_set_name, 'labels-test', dataBasePath)

X_val = load_data(data_set_name, 'features-valid', dataBasePath)
Y_val = load_data(data_set_name, 'labels-valid', dataBasePath)

from scratch import to_one_hot
Y_train_OH = to_one_hot(Y_train)
Y_test_OH = to_one_hot(Y_test)
Y_val_OH = to_one_hot(Y_val)

print('X_train:', np.shape(X_train))
print('Y_train:', np.shape(Y_train))
print('Y_train_OH:', np.shape(Y_train_OH))
print('X_test:', np.shape(X_test))
print('Y_test:', np.shape(Y_test))
print('Y_test_OH:', np.shape(Y_test_OH))
print('X_val:', np.shape(X_val))
print('Y_val:', np.shape(Y_val))
print('Y_val_OH:', np.shape(Y_val_OH))

X = np.reshape(X_train, (-1, 21))
Y = np.reshape(Y_train, (-1))
# P = compute_joint_probabilities(X, batch_size=batch_size, verbose=2)
# save_data(P, data_set_name, 'joint_P', dataBasePath)
P = load_data(data_set_name, 'joint_P', dataBasePath)

# no_dims=2, perplexity=50, theta=0.5, randseed=-1, verbose=False,
# initial_dims=50, use_pca=True, max_iter=1000,

# X_2d = run_bh_tsne(X, no_dims=numDims, initial_dims=initialDims,
#                    verbose=True, perplexity=perplexity, theta=theta,
#                    usefile=False, array=X)

# save_data(X_2d, data_set_name, 'X_2d', dataBasePath)
from tsne import bh_sne
# X_2d = bh_sne(X_train.astype(np.float64))
# save_data(X_2d, data_set_name, 'X_2d_new', dataBasePath)

X_2d = load_data(data_set_name, 'X_2d_new', dataBasePath)

print('x_2d shape:', np.shape(X_2d))
# print('x_2d_old shape:', np.shape(X_2d_old))

# from tsne import bh_sne
# X_2d = bh_sne(X_train.astype(np.float64))

plot_model(X_2d, Y_train)
plt.savefig('X_2d.png')


def get_tb_cb(modelName):
    return TensorBoard(log_dir='/tmp/tflearn_logs/keras-' + modelName + '-' + date.today().isoformat() + '-' + str(time.time()), histogram_freq=10, write_graph=True, write_images=False)


# Build encoder and decoder
encoding_dim = 10
input_dim = X_train.shape[1]

inputs = Input(shape=(input_dim,))

x = Dense(10, activation='relu', name='encoder_0_L_1')(inputs)
x = Dense(32, activation='relu', name='encoder_0_L_2')(x)
encoded = Dense(encoding_dim, activation='relu', name='encoder_0_L_3_encode')(x)
x = Dense(32, activation='relu', name='decoder_0_L_1')(encoded)
x = Dense(10, activation='relu', name='decoder_0_L_2')(x)
decoded = Dense(X_train.shape[1], activation='relu', name='decoder_0_L_3_decode')(x)

encoder_model = Model(input=inputs, output=encoded)

autoencoder_model = Model(input=inputs, output=decoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder_model.layers[-1]
decoder_model = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder_model.compile(loss='mse', optimizer='rmsprop')
autoencoder_model.fit(X_train, X_train, nb_epoch=100, verbose=2, callbacks=[get_tb_cb('vanilla_encoder')]) #, validation_data=(X_test, X_test)

yaml_string = encoder_model.to_yaml()

f = open(dataBasePath + data_set_name +'/vanilla_encoder_arch.yaml', 'w')
f.write(yaml_string)
f.close()

encoder_model.save_weights(dataBasePath + data_set_name + '/vanilla_encoder_weights.keras')

plot_model(autoencoder_model.predict(X_train), Y_train)
plt.savefig('vanilla_encoder_train.png')
plot_model(autoencoder_model.predict(X_test), Y_test)
plt.savefig('vanilla_encoder_train.png')
plot_differences(autoencoder_model.predict(X_train), X_train)
plt.savefig('vanilla_encoder_train_diff.png')
plot_differences(autoencoder_model.predict(X_test), X_test)
plt.savefig('vanilla_encoder_test_diff.png')

chk = ModelCheckpoint('vanilla_autoencoder', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

encoded.trainable = False

print('input_dim:', input_dim)
input2 = Input(shape=(input_dim,))

# x = Dense(10, activation='relu', name='ft_encoder_0_L_1', weights=encoder_model.layers[1].get_weights())(input2)
# x = Dense(32, activation='relu', name='ft_encoder_0_L_2', weights=encoder_model.layers[2].get_weights())(x)
# encoded = Dense(encoding_dim, activation='relu', name='ft_encoder_0_L_3_encode', weights=encoder_model.layers[3].get_weights())(x)
x = Dense(512, activation='relu', name='mlp_0_L_1')(encoded)
x = Dense(1024, activation='relu', name='mlp_0_L_2')(x)
x = Dense(256, activation='relu', name='mlp_0_L_3')(x)
out = Dense(2, activation='softmax' )(x)
sgd = SGD(lr=0.1)
finetune_model = Model(input=inputs, output=out)
finetune_model.compile(loss='categorical_crossentropy', optimizer=sgd)
finetune_model.fit(X_train, Y_train_OH, validation_split=0.1, nb_epoch=100, verbose=2)#, callbacks=[get_tb_cb('vanilla_ae_finetune')])
# , validation_data=(X_test, Y_test_OH)
# model = Sequential()
# model.add(Dense(500, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(2000, activation='relu'))
# model.add(Dense(2))
# sgd = SGD(lr=0.1)
# model.compile(loss=tsne, optimizer=sgd)

# print('P.shape:', P.shape)
# P_train = P.reshape(X_train.shape[0], -1)
# print('P.shape:', P.shape)
# print('X_train.shape:', X_train.shape)
# print('Y_train.shape:', Y_train.shape)

# model.fit(X_train, P_train, validation_split=0.1, batch_size=batch_size, shuffle=False, nb_epoch=200, callbacks=[get_tb_cb('tsne-ae')])

# plot_model(model.predict(X_train), Y_train)
# plt.savefig('tsne-ae-train.png')
# plot_model(model.predict(X_test), Y_test)
# plt.savefig('tsne-ae-test.png')


def old_stuff():
    encoder = Sequential()
    encoder.add(Dense(500, activation='relu', input_shape=(X_train.shape[1],)))
    encoder.add(Dense(500, activation='relu'))
    encoder.add(Dense(2000, activation='relu'))
    encoder.add(Dense(2))
    encoder.compile(loss='mse', optimizer='rmsprop')

    # plot(encoder, to_file='encoder.png')

    encoder.fit(X_train, X_2d, nb_epoch=200, verbose=2, callbacks=[get_tb_cb('encoder')])

    plot_model(encoder.predict(X_train), Y_train)
    plt.savefig('encoder-train.png')
    plot_model(encoder.predict(X_test), Y_test)
    plt.savefig('encoder-test.png')
    plot_differences(encoder.predict(X_train), X_2d)
    plt.savefig('encoder-train-diff.png')
    plot_differences(encoder.predict(X_test), X_2d)
    plt.savefig('encoder-test-diff.png')

    decoder = Sequential()
    decoder.add(Dense(2000, activation='relu', input_shape=(2,)))
    decoder.add(Dense(500, activation='relu'))
    decoder.add(Dense(500, activation='relu'))
    decoder.add(Dense(X_train.shape[1]))
    decoder.compile(loss='mse', optimizer='rmsprop')

    # plot(decoder, to_file='decoder.png')

    decoder.fit(X_2d, X_train, nb_epoch=100, verbose=2, callbacks=[get_tb_cb('decoder')])


    n = X_train.shape[1]
    ae = Sequential()
    ae.add(Dense(500, activation='relu', weights=encoder.layers[0].get_weights(), input_shape=(n, )))
    ae.add(Dense(500, activation='relu', weights=encoder.layers[1].get_weights()))
    ae.add(Dense(2000, activation='relu', weights=encoder.layers[2].get_weights()))
    ae.add(Dense(2, weights=encoder.layers[3].get_weights()))
    ae.add(Dense(2000, activation='relu', weights=decoder.layers[0].get_weights()))
    ae.add(Dense(500, activation='relu', weights=decoder.layers[1].get_weights()))
    ae.add(Dense(500, activation='relu', weights=decoder.layers[2].get_weights()))
    ae.add(Dense(n, weights=decoder.layers[3].get_weights()))
    ae.compile(loss='mse', optimizer='rmsprop')

    # plot(ae, to_file='ae.png')

    ae.fit(X_train, X_train, validation_data=(X_test, X_test), nb_epoch=100, verbose=2, batch_size=32, callbacks=[get_tb_cb('ae')])

    decoded = ae.predict(X_train)
    plot_model(decoded, Y_train)
    plt.savefig('decoded-train.png')
    encoded = encoder.predict(X_train)
    plot_model(encoded, Y_train)
    plt.savefig('encoded-ytrain.png')

    #Build MLP using ae_encoder half


    ae_encoder = Sequential()
    ae_encoder.add(Dense(2000, activation='relu', weights=ae.layers[0].get_weights(), input_shape=(2, )))
    ae_encoder.add(Dense(500, activation='relu', weights=ae.layers[1].get_weights()))
    ae_encoder.add(Dense(500, activation='relu', weights=ae.layers[2].get_weights()))
    ae_encoder.add(Dense(X_train.shape[1], weights=ae.layers[3].get_weights()))
    ae_encoder.compile(loss='mse', optimizer='rmsprop')

    yaml_string = ae_encoder.to_yaml()
    f = open('workfile', 'r+')
    f.write(yaml_string)
    f.close()

    ae_encoder.save_weights(dataBasePath + '/' + data_set_name + '/ae_encoder_weights.keras')

    ae_encoded = ae_encoder.predict(X_train)

    # plot(ae_encoder, to_file='ae_encoder.png')
