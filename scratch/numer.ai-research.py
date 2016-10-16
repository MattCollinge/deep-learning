# from bhtsne import run_bh_tsne
import numpy as np
import keras
from keras import backend as K
from keras import objectives
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from datetime import date
from sklearn import preprocessing

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
batch_size = 8000
pretrain = True

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

X_tourn = load_data(data_set_name, 'tournament', dataBasePath)
X_tourn = np.reshape(X_train, (-1, 21))

X_test = load_data(data_set_name, 'features-test', dataBasePath)
Y_test = load_data(data_set_name, 'labels-test', dataBasePath)

X_val = load_data(data_set_name, 'features-valid', dataBasePath)
Y_val = load_data(data_set_name, 'labels-valid', dataBasePath)

X_Test_Unsup = load_data(data_set_name, 'unsup-features-test', dataBasePath)
X_Train_Unsup = load_data(data_set_name, 'unsup-features-train', dataBasePath)


from scratch import to_one_hot
Y_train_OH = to_one_hot(Y_train)
Y_test_OH = to_one_hot(Y_test)
Y_val_OH = to_one_hot(Y_val)

print('X_train:', np.shape(X_train))
print('Y_train:', np.shape(Y_train))
print('X_tourn:', np.shape(X_tourn))

print('X_Train_Unsup:', np.shape(X_Train_Unsup))
print('Y_train_OH:', np.shape(Y_train_OH))
print('X_test:', np.shape(X_test))
print('Y_test:', np.shape(Y_test))
print('Y_test_OH:', np.shape(Y_test_OH))
print('X_val:', np.shape(X_val))
print('Y_val:', np.shape(Y_val))
print('Y_val_OH:', np.shape(Y_val_OH))



# X = np.reshape(X_Train_Unsup, (-1, 21))
# Y = np.reshape(Y_train, (-1))
# P = compute_joint_probabilities(X, batch_size=batch_size, verbose=2)
# save_data(P, data_set_name, 'joint_P', dataBasePath)
# P = load_data(data_set_name, 'joint_P', dataBasePath)

# no_dims=2, perplexity=50, theta=0.5, randseed=-1, verbose=False,
# initial_dims=50, use_pca=True, max_iter=1000,

# X_2d = run_bh_tsne(X, no_dims=numDims, initial_dims=initialDims,
#                    verbose=True, perplexity=perplexity, theta=theta,
#                    usefile=False, array=X)

# save_data(X_2d, data_set_name, 'X_2d', dataBasePath)
from tsne import bh_sne
# X_2d_trn = bh_sne(X_train.astype(np.float64))
# X_2d_tst = bh_sne(X_test.astype(np.float64))
# X_2d_u = bh_sne(X_Train_Unsup.astype(np.float64))
# X_2d_u_tst = bh_sne(X_Test_Unsup.astype(np.float64))

# save_data(X_2d_trn, data_set_name, 'X_2d_trn', dataBasePath)
# save_data(X_2d_tst, data_set_name, 'X_2d_tst', dataBasePath)
# save_data(X_2d_u_trn, data_set_name, 'X_2d_u_trn', dataBasePath)
# save_data(X_2d_u_tst, data_set_name, 'X_2d_u_tst', dataBasePath)

scaler = preprocessing.maxabs_scale #MinMaxScaler().fit_transform #maxabs_scale

X_2d_u_trn = scaler(load_data(data_set_name, 'X_2d_u_trn', dataBasePath))
X_2d_u_tst = scaler(load_data(data_set_name, 'X_2d_u_tst', dataBasePath))
X_2d_trn = scaler(load_data(data_set_name, 'X_2d_trn', dataBasePath))
X_2d_tst = scaler(load_data(data_set_name, 'X_2d_tst', dataBasePath))


print('X_2d_u shape:', np.shape(X_2d_u_trn))
print('X_2d_trn shape:', np.shape(X_2d_trn))
print('X_2d_tst shape:', np.shape(X_2d_tst))

# X_Train_Unsup = np.concatenate((X_Train_Unsup, X_2d_u_trn), axis=1)
# X_Test_Unsup = np.concatenate((X_Test_Unsup, X_2d_u_tst), axis=1)
# X_train = np.concatenate((X_train, X_2d_trn), axis=1)
# X_test = np.concatenate((X_test, X_2d_tst), axis=1)

print('X_train:', np.shape(X_train))
print('X_Train_Unsup:', np.shape(X_Train_Unsup))
print('X_test:', np.shape(X_test))

plot_model(X_2d_trn, Y_train)
plt.savefig('X_2d_trn.png')

plot_model(X_2d_tst, Y_test)
plt.savefig('X_2d_tst.png')


def get_tb_cb(modelName):
    run_name = 'keras-' + modelName + '-' + date.today().isoformat() + '-' + str(time.time())
    print('Cutting TB run: ', run_name)
    return TensorBoard(log_dir='/tmp/tflearn_logs/' + run_name, histogram_freq=0, write_graph=True, write_images=False)


def plot_outcomes(model, model_name, X_train, X_test, Y_train, Y_test):
        plt.close('all')
        plot_model(model.predict(X_train), Y_train)
        plt.savefig(model_name + '_train.png')
        plot_model(model.predict(X_test), Y_test)
        plt.savefig(model_name + '_test.png')

        plot_differences(model.predict(X_train), X_train)
        plt.savefig(model_name + 'train_diff.png')
        plot_differences(model.predict(X_test), X_test)
        plt.savefig(model_name + 'test_diff.png')


chk = ModelCheckpoint(dataBasePath + data_set_name + '/ae_finetune.ckpt', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')


def pretrain_ae(ae_model, encoder_model, nb_epoch, ae_name, ae_batch_size, X_train, X_test, weights_file):
    
    ae_model.fit(X_train, X_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=ae_batch_size,
        validation_data=(X_test, X_test),
        callbacks=[get_tb_cb(ae_name)])

    encoder_model.save_weights(weights_file)

    # plot_outcomes(vae_model, 'vae' + vae_name, X_train, X_test, Y_train, Y_test)
    
    next_X_train = encoder_model.predict(X_train, batch_size=ae_batch_size)
    next_X_test = encoder_model.predict(X_test, batch_size=ae_batch_size)

    # abs_scalar = preprocessing.MaxAbsScaler().fit_transform

    # x_tst = abs_scalar(next_X_test) # min_max_scaler.fit_transform(std_scaler.fit_transform(next_X_test))
    # x_trn = abs_scalar(next_X_train) # min_max_scaler.fit_transform(std_scaler.fit_transform(next_X_train))

    # std = np.std(x_tst, dtype=np.float64)
    # mean = np.mean(x_tst, dtype=np.float64)

    # print ('x_tst: Std=' + str(std), 'Mean=' + str(mean))


    min_max_scaler = preprocessing.MinMaxScaler() #xStandardScaler()
    # std_scaler = preprocessing.StandardScaler()
    next_X_train = min_max_scaler.fit_transform(next_X_train)
    next_X_test = min_max_scaler.fit_transform(next_X_test)

    return next_X_train, next_X_test


def create_vae(start_layer, depth, latent_dim, intermediate_dim, x_shape, name_suffix, trainable=True):
        batch_sample_mode = 2
        start_layer = BatchNormalization(mode=batch_sample_mode, name='encoder_{0}_L_start_BatchNorm'.format(name_suffix))(start_layer)
        
        h = Dense(intermediate_dim, activation='relu', name='encoder_{0}_L_1'.format(name_suffix), trainable=trainable)(start_layer)
        z_mean = Dense(latent_dim, name='encoder_{0}_L_2_encode'.format(name_suffix), trainable=trainable)(h)
        # z_mean = BatchNormalization(mode=batch_sample_mode, name='encoder_{0}_L_z_mean_BatchNorm'.format(name_suffix))(z_mean)
        
        z_log_var = Dense(latent_dim, name='encoder_{0}_L_3'.format(name_suffix), trainable=trainable)(h)
        # z_log_var = BatchNormalization(mode=batch_sample_mode, name='encoder_{0}_L_z_log_var_BatchNorm'.format(name_suffix))(z_log_var)
        
        original_dim = x_shape

        def sampling(args):
            epsilon_std = 0.01
            z_mean, z_log_var = args
            epsilon = K.random_normal(K.shape(z_mean), mean=0., #shape=(vae_batch_size, latent_dim), mean=0.,
                                  std=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon


        def kullback_leibler_divergence(y_true, y_pred):
            y_true = K.clip(y_true, K.epsilon(), 1)
            y_pred = K.clip(y_pred, K.epsilon(), 1)
            return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
            # z_mean_c = K.clip(z_mean, K.epsilon(), 1)
            # z_log_var_c = K.clip(z_log_var, K.epsilon(), 1)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss


        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=[latent_dim,], name='encoder_{0}_L_Lambda'.format(name_suffix))([z_mean, z_log_var])
        
        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_{0}_L_1'.format(name_suffix))
        decoder_mean = Dense(original_dim, activation='sigmoid', name='decoder_{0}_L_2'.format(name_suffix))
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        return x_decoded_mean, z, vae_loss


def train_vae_model():
    
    num_layers = 3
    vae_batch_size = 5000
    latent_dim = 15
    intermediate_dim = 1000
    nb_epoch = 150


    pretrained_layers = []
    vae_layers = []
    pretrained_weights = {}
    x_tst = X_Test_Unsup #X_test
    x_trn = X_Train_Unsup #X_train

    rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.0001, beta_1=0.8, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    
    for vae in range(num_layers):
        vae_name = 'vae_{0}'.format(vae)
        weights_file = dataBasePath + data_set_name + '/vae_' + vae_name + '_weights.keras'

        input_dim = x_trn.shape[1]
        print('input_dim:', input_dim, '. Full array:', np.shape(x_trn))

        x = Input(shape=(input_dim,), name='input_{0}_L'.format(vae_name))

        # Build one vae
        x_decoded_mean, z, vae_loss = create_vae(x, vae, latent_dim, intermediate_dim, input_dim, vae_name, True) #- (num_layers * 10)

        encoder_model = Model(input=x, output=z)
        vae_model = Model(input=x, output=x_decoded_mean)

        # print('vae', vae_name, '.layers: ', encoder_model.layers)
        print('vae', vae_name, '.summary: ', encoder_model.summary())
        # print('vae', vae_name, '.config: ', encoder_model.get_config())
    
        # config = encoder_model.get_config()
        
        # store layers in array
        for l in encoder_model.layers[1:]:
            pretrained_layers.append(l)

        vae_layers.append(encoder_model)

        vae_model.compile(optimizer=nadam, loss=vae_loss)

        layer_epoch = nb_epoch
        # if(vae != 0):
        #     layer_epoch = 10

        # train it
        x_trn, x_tst = pretrain_ae(vae_model, encoder_model, layer_epoch, 'vae_encoder_' + vae_name, vae_batch_size, x_trn, x_tst, weights_file)

        # store weights in dict by layer name
        for l in encoder_model.layers:
            pretrained_weights[l.name] = l.get_weights()

    # rebuild model from layers
    # finetune_model = Sequential()
    vae_name = 'vae_{0}'.format(0)
    input_dim = X_train.shape[1] 
    x = Input(shape=(input_dim,), name='input_{0}_L'.format(vae_name))
    # _, y, _ = create_vae(x, 0, latent_dim, intermediate_dim, input_dim, vae_name, True) #- (num_layers * 10)
    y = x
    for vae in range(num_layers):
       vae_name = 'vae_{0}'.format(vae)
       _, y, _ = create_vae(y, vae, latent_dim, intermediate_dim, input_dim, vae_name, True) #- (num_layers * 10)
      

    pretrained_model = Model(input=x, output=y)

     
    for l in pretrained_model.layers:
        print(l.name)
        l.set_weights(pretrained_weights[l.name])
        # finetune_model.add(l)

    finetune_model = pretrained_model
    batch_sample_mode = 2
    #y = Dropout(0.5)(y)
    y = BatchNormalization(mode=batch_sample_mode)(y)
        
    out = Dense(2, activation='softmax')(y)
    # finetune_model.add(d)
    # finetune_model.add(out)
    finetune_model = Model(input=x, output=out)
    
    sgd = SGD(lr=0.01, decay=1e-15) #, nesterov=True )
    nadamft = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    # print('Finetune_model.layers: ', finetune_model.layers)
    print('Finetune_model.summary: ', finetune_model.summary())
    # print('Finetune_model.config: ', finetune_model.get_config())
    # finetune_model.load_weights(weights_file, by_name=True)

    run_model = 'vae_ae_finetune' #'vanilla_ae_finetune' # 'vae_ae_finetune'

    finetune_model.compile(loss='categorical_crossentropy', optimizer=sgd)
    finetune_model.fit(X_train, Y_train_OH, validation_data=(X_test, Y_test_OH), batch_size=batch_size, nb_epoch=200, verbose=2, callbacks=[chk, get_tb_cb(run_model)])

def train_ae_model():
    
    num_layers = 2
    pretrained_layers = []
    pretrained_weights = {}
    x_tst = X_Test_Unsup #X_test
    x_trn = X_Train_Unsup #X_train
    
    latent_dim = 4
    ae_batch_size = batch_size
    
    for ae in range(num_layers):
        ae_name = 'ae_{0}'.format(ae)
        weights_file = dataBasePath + data_set_name + '/ae_' + ae_name + '_weights.keras'

        input_dim = x_trn.shape[1]
        print('input_dim:', input_dim, '. Full array:', np.shape(x_trn))

        x = Input(shape=(input_dim,))

        # Build one vae
        encoded, decoded  = create_auto_encoder(x, latent_dim, input_dim, ae_name, True) #- (num_layers * 10)

        encoder_model = Model(input=x, output=encoded)
        ae_model = Model(input=x, output=decoded)

        # store layers in array
        for l in encoder_model.layers[1:]:
            pretrained_layers.append(l)

        rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ae_model.compile(optimizer=adam, loss='mse')

        # train it
        x_trn, x_tst = pretrain_ae(ae_model, encoder_model, 'ae_encoder_' + ae_name, ae_batch_size, x_trn, x_tst, weights_file)

        # store weights in dict by layer name
        for l in encoder_model.layers[1:]:
            pretrained_weights[l.name] = l.get_weights()

    # rebuild model from layers
    finetune_model = Sequential()
    for l in pretrained_layers:
        l.set_weights(pretrained_weights[l.name])
        finetune_model.add(l)

    out = Dense(2, activation='softmax')
    finetune_model.add(out)

    sgd = SGD(lr=0.1, decay=1e-15) #, nesterov=True )

    print('Finetune_model.layers: ', finetune_model.layers)
    print('Finetune_model.summary: ', finetune_model.summary())
    print('Finetune_model.config: ', finetune_model.get_config())
    finetune_model.load_weights(weights_file, by_name=True)

    run_model = 'ae_ae_finetune' #'vanilla_ae_finetune' # 'vae_ae_finetune'

    finetune_model.compile(loss='categorical_crossentropy', optimizer=sgd)
    finetune_model.fit(X_train, Y_train_OH, validation_data=(X_test, Y_test_OH), batch_size=batch_size, nb_epoch=700, verbose=2, callbacks=[chk, get_tb_cb(run_model)])



def vae(x, input_dim, pretrain, trainable):

    num_layers = 4
    vae_batch_size = 5000
    latent_dim = 500
    intermediate_dim = 3000
    nb_epoch = 20
        
    trained_layers = x

    def create_vae(start_layer, latent_dim, intermediate_dim, x_shape, name_suffix, trainable):
        batch_sample_mode = 2
        # start_layer = BatchNormalization(mode=batch_sample_mode, name='encoder_{0}_L_start_BatchNorm'.format(name_suffix))(start_layer)
        # init='he_normal',
        h = Dense(intermediate_dim, activation='relu', name='encoder_{0}_L_1'.format(name_suffix), trainable=trainable)(start_layer)
        z_mean = Dense(latent_dim, name='encoder_{0}_L_2_encode'.format(name_suffix), trainable=trainable)(h)
        z_log_var = Dense(latent_dim, name='encoder_{0}_L_3'.format(name_suffix), trainable=trainable)(h)

        original_dim = x_shape

        def sampling(args):
            epsilon_std = 0.01
            z_mean, z_log_var = args
            epsilon = K.random_normal(K.shape(z_mean), mean=0., #shape=(vae_batch_size, latent_dim), mean=0.,
                                  std=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon


        def vae_loss(x, x_decoded_mean):
            xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss


        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='encoder_{0}_L_Lambda'.format(name_suffix))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_{0}_L_1'.format(name_suffix))
        decoder_mean = Dense(original_dim, activation='sigmoid', name='decoder_{0}_L_2'.format(name_suffix))
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        return x_decoded_mean, z, vae_loss


    for vae in range(num_layers):
        vae_name = 'vae_{0}'.format(vae)

        x_decoded_mean, z_mean, vae_loss = create_vae(trained_layers, latent_dim, intermediate_dim, input_dim, vae_name, trainable) #- (num_layers * 10)

        encoder_model = Model(input=x, output=z_mean)
        vae_model = Model(input=x, output=x_decoded_mean)

        weights_file = dataBasePath + data_set_name + '/vae_' + vae_name + '_weights.keras'

        if(pretrain==True):
            rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
            nadam = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
            vae_model.compile(optimizer=nadam, loss=vae_loss)

            # vae_model.load_weights(weights_file, by_name=True)

            print('vae_model.summary: ', vae_model.summary())
            # print('vae_model.config: ', vae_model.get_config())

            print('encoder_model.summary: ', encoder_model.summary())
            print('encoder_model.config: ', vae_model.get_config())

            vae_model.fit(X_Train_Unsup, X_Train_Unsup,
                shuffle=True,
                nb_epoch=nb_epoch,
                batch_size=vae_batch_size,
                validation_data=(X_Test_Unsup, X_Test_Unsup),
                callbacks=[get_tb_cb('vae_encoder_' + vae_name)])

            # build some new layers and assign the trained weights but freeze them??

            yaml_string = encoder_model.to_yaml()

            f = open(dataBasePath + data_set_name +'/vae_arch_' + vae_name + '.yaml', 'w')
            f.write(yaml_string)
            f.close()

            encoder_model.save_weights(weights_file)

            for l in encoder_model.layers:
                l.trainable = False

            encoder_model.compile(optimizer=nadam, loss=vae_loss)
            # x_decoded_mean, z_mean, vae_loss = create_vae(trained_layers, latent_dim, intermediate_dim, input_dim, vae_name, False) #- (num_layers * 10)
            #
            # encoder_model = Model(input=x, output=z_mean)
            # # vae_model = Model(input=x, output=x_decoded_mean)
            #
            # # vae_model.load_weights(weights_file, by_name=True)
            # encoder_model.load_weights(weights_file, by_name=True)

            # plot_outcomes(vae_model, 'vae' + vae_name, X_train, X_test, Y_train, Y_test)

        trained_layers = z_mean

    return weights_file, z_mean



# AutoEncoder

def create_auto_encoder(start_layer, encoding_dim, x_shape, name_suffix, trainable):
    x = Dense(256, activation='relu', name='encoder_{0}_L_1'.format(name_suffix), trainable=trainable)(start_layer)
    x = Dense(128, activation='relu', name='encoder_{0}_L_2'.format(name_suffix), trainable=trainable)(x)
    encoded = Dense(encoding_dim, activation='relu', name='encoder_{0}_L_3_encode'.format(name_suffix), trainable=trainable)(x)
    x = Dense(128, activation='relu', name='decoder_{0}_L_1'.format(name_suffix), trainable=trainable)(encoded)
    x = Dense(256, activation='relu', name='decoder_{0}_L_2'.format(name_suffix), trainable=trainable)(x)
    decoded = Dense(x_shape, activation='relu', name='decoder_{0}_L_3_decode'.format(name_suffix), trainable=trainable)(x)
    return encoded, decoded


def auto_encoder(inputs, pretrain, trainable):
    # Build encoder and decoder
    encoding_dim = 100
    num_layers = 5
    trained_layers = inputs

    for ae in range(num_layers):
        ae_name = 'ae_{0}'.format(ae)

        encoded, decoded = create_auto_encoder(trained_layers, encoding_dim - (num_layers * 10), input_dim, ae_name, trainable) #- (num_layers * 10)
        encoder_model = Model(input=inputs, output=encoded)
        autoencoder_model = Model(input=inputs, output=decoded)
        print('encoder_model.summary: ', encoder_model.summary())
        print('autoencoder_model.summary: ', autoencoder_model.summary())
        trained_layers = encoded
        weights_file = dataBasePath + data_set_name + '/vanilla_encoder' + ae_name + '_weights.keras'

        if(pretrain==True):
            autoencoder_model.compile(loss='mse', optimizer='rmsprop')
            autoencoder_model.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=batch_size, nb_epoch=100, verbose=2, callbacks=[get_tb_cb('vanilla_encoder_' + ae_name)]) #, validation_data=(X_test, X_test)

            # trained_layers.trainable = False

            yaml_string = encoder_model.to_yaml()

            f = open(dataBasePath + data_set_name +'/vanilla_encoder_arch' + ae_name + '.yaml', 'w')
            f.write(yaml_string)
            f.close()

            encoder_model.save_weights(weights_file)

            plot_outcomes(autoencoder_model, 'ae' + vae_name, X_test, X_train, Y_test, Y_train)


    return weights_file, encoded


def run():
    input_dim = X_train.shape[1]
    print('input_dim:', input_dim)
    batch_sample_mode = 2

    inputs = Input(shape=(input_dim,))
    trainable=True

    weights_file, encoded =  vae(inputs, input_dim, pretrain, trainable) #vae(inputs)  #auto_encoder(inputs)

    # encoded = BatchNormalization(mode=batch_sample_mode)(encoded)
    # x = Dropout(0.5)(encoded)
    out = Dense(2, activation='softmax')(encoded)
    sgd = SGD(lr=0.01, decay=1e-15, nesterov=True )
    nadamft = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    finetune_model = Model(input=inputs, output=out)

    for l in finetune_model.layers:
        l.trainable = True

    if(pretrain==False):
        print('Loading weights from: ', weights_file)
        print('Finetune_model.layers: ', finetune_model.layers)
        print('Finetune_model.summary: ', finetune_model.summary())
        print('Finetune_model.config: ', finetune_model.get_config())
        finetune_model.load_weights(weights_file, by_name=True)

    print('Finetune_model.summary: ', finetune_model.summary())

    run_model = 'vae_ae_finetune' #'vanilla_ae_finetune' # 'vae_ae_finetune'

    finetune_model.compile(loss='categorical_crossentropy', optimizer=sgd)
    finetune_model.fit(X_train, Y_train_OH, validation_data=(X_test, Y_test_OH), batch_size=batch_size, nb_epoch=700, verbose=2, callbacks=[chk, get_tb_cb(run_model)])

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

# train_ae_model()
# train_vae_model()
run()

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
