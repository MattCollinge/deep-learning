from bhtsne import run_bh_tsne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
# %matplotlib inline

plt.close('all')

numDims = 2
initialDims = 21
perplexity = 30
theta = .5
alg = 'svd'

def load_data(data_set_name, data_set_type):
    dataBasePath = '/home/dev/data/numer.ai/' + data_set_name + '/'
    return np.load(dataBasePath + data_set_type + '-' + data_set_name + '.npy')


data_set_name = '2016-09-08'

X = load_data(data_set_name, 'features')
Y = load_data(data_set_name, 'labels')

X = np.reshape(X, (-1, 21))
Y = np.reshape(Y, (-1))
print(np.shape(X))
print(np.shape(Y))

# no_dims=2, perplexity=50, theta=0.5, randseed=-1, verbose=False,
# initial_dims=50, use_pca=True, max_iter=1000,
map = run_bh_tsne(X[0:10000], no_dims=numDims, initial_dims=initialDims,
                  verbose=True, perplexity=perplexity, theta=theta,
                  usefile=False, array=X[0:10000])
# gscatter(map(:,1), map(:,2), Y);

plt.scatter(map[:, 0], map[:, 1], 20, ('b', 'g'))
plt.show()
plt.savefig('bhtsne.png')
