import numpy as np
import matplotlib.pyplot as plt
import png
import os
import random
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib

matplotlib.use('Agg')
# %matplotlib inline
import matplotlib.pyplot as plt
plt.close('all')

dataBasePath = '/home/dev/data/numer.ai/2016-09-08/'
csvfilename = dataBasePath + 'numerai_training_data.csv'

# data = np.loadtxt(csvfilename, skiprows=1, usecols=range(0, 22), delimiter=',', dtype='f8')
# df = pd.DataFrame(data)
df = pd.read_csv(csvfilename)
maxrows = 150000

#print(df.head())
def summaryInfo(data):
#     for column in data:
#         plt.figure(figsize=(20,4))
#         plt.suptitle(column)
#         plt.hist(data[column], bins=30, range=[0,1]);
#         plt.savefig(column+ '.png')
    print(data.describe())
# summaryInfo(df)

# plt.plot(df.iloc[1000:1100,1], 'r-')

features = df.values[:,0:21]
labels = df.values[:,21]

print(features[0])
print(labels[0])

print(np.shape(features))

features = np.reshape(features, [-1, 21, 1])

print(np.shape(features))

np.save(dataBasePath + 'features-2016-09-08.npy', features)

std = np.std(features, dtype=np.float64)
mean = np.mean(features, dtype=np.float64)
print ('features: Std=' + str(std), 'Mean=' + str(mean))

np.save(dataBasePath + 'labels-2016-09-08.npy', labels)
