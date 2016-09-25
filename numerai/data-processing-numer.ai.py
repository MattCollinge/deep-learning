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

validation_pc = 0.1
test_pc = 0.1
data_set_name = '2016-09-08'
dataBasePath = '/home/dev/data/numer.ai/' + data_set_name + '/'
csvfilename = dataBasePath + 'numerai_training_data.csv'
tournamentcsvfilename = dataBasePath + 'numerai_tournament_data.csv'

# data = np.loadtxt(csvfilename, skiprows=1, usecols=range(0, 22), delimiter=',', dtype='f8')
# df = pd.DataFrame(data)
df = pd.read_csv(csvfilename)
dft = pd.read_csv(tournamentcsvfilename)
# maxrows = 150000

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
tournament = dft.values[:,1:22]
tournament_ids = pd.DataFrame(dft.values[:,0],dtype=int).values

features = df.values[:,0:21]
labels = df.values[:,21]

# for non tfLearn uses - create train, validation and test sets
dataset_size = len(df.values)
validation_count = int(dataset_size * validation_pc)
test_count = int(dataset_size * test_pc)

df = df.sample(frac=1)

features_test = df.values[0:test_count,0:21]
labels_test = df.values[0:test_count,21]

features_valid = df.values[test_count:(test_count + validation_count),0:21]
labels_valid = df.values[test_count: (test_count + validation_count),21]


features_train = df.values[(test_count + validation_count):, 0:21]
labels_train = df.values[(test_count + validation_count):, 21]



print(features[0])
print(labels[0])
print(tournament[0])
print(tournament_ids[0])

print('features:', np.shape(features))
print('labels:',np.shape(labels))
print('tournament:', np.shape(tournament))
print('tournament_ids:',np.shape(tournament_ids))

features = np.reshape(features, [-1, 21, 1])
tournament = np.reshape(tournament, [-1, 21, 1])

print(np.shape(features))

np.save(dataBasePath + 'tournament-' + data_set_name + '.npy', tournament)
np.save(dataBasePath + 'tournament_ids-' + data_set_name + '.npy', tournament_ids)

np.save(dataBasePath + 'features-' + data_set_name + '.npy', features)
np.save(dataBasePath + 'labels-' + data_set_name + '.npy', labels)

np.save(dataBasePath + 'features-train-' + data_set_name + '.npy', features_train)
np.save(dataBasePath + 'labels-train-' + data_set_name + '.npy', labels_train)
np.save(dataBasePath + 'features-valid-' + data_set_name + '.npy', features_valid)
np.save(dataBasePath + 'labels-valid-' + data_set_name + '.npy', labels_valid)
np.save(dataBasePath + 'features-test-' + data_set_name + '.npy', features_test)
np.save(dataBasePath + 'labels-test-' + data_set_name + '.npy', labels_test)

std = np.std(features, dtype=np.float64)
mean = np.mean(features, dtype=np.float64)
print ('features: Std=' + str(std), 'Mean=' + str(mean))

std = np.std(tournament, dtype=np.float64)
mean = np.mean(tournament, dtype=np.float64)
print ('tournament: Std=' + str(std), 'Mean=' + str(mean))
