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

validation_pc = 0.15
test_pc = 0.1
data_set_name = 'numerai_datasets_22-23-24' #2016-09-08'
dataBasePath = '/home/dev/data/numer.ai/'
csvfilename = 'numerai_training_data.csv'
tournamentcsvfilename =  'numerai_tournament_data.csv'

# Make BIG file with all data in
def do_all_data_sets(data_set_path):
    start = 22
    end = 25
    for i in range(start, end):
        path = data_set_path + 'numerai_datasets_{0}/'.format(i)
        print('loading data from {0}'.format(path))
        dfi = pd.read_csv(path + csvfilename)
        dfti = pd.read_csv(path + tournamentcsvfilename)
        summaryInfo(dfi, path + csvfilename)
        summaryInfo(dfti, path + tournamentcsvfilename)
        if i == start:
            df = dfi
            dft = dfti
        else:
            df = pd.concat([df, dfi])
            dft = pd.concat([dft, dfti])
        summaryInfo(df, 'Cumulative')
        summaryInfo(dft, 'Cumulative Tournament')
    return df, dft


def do_single_data_set(data_set_path):

    df = pd.read_csv(data_set_path + csvfilename)
    dft = pd.read_csv(data_set_path + tournamentcsvfilename)

    return df, dft
    # maxrows = 150000

def dedupe(df):
   dupes = df.drop_duplicates()
   return dupes


#print(df.head())
def summaryInfo(data, data_set_name):
    # for column in data:
    #     plt.figure(figsize=(20,4))
    #     plt.suptitle(column)
    #     plt.hist(data[column], bins=30, range=[0,1]);
    #     plt.savefig(column+ '.png')
    # print(data.describe())
    print("Count ({0}): {1}".format(data_set_name, data.shape))
# summaryInfo(df)


def prep_datasets(show_stats):

    data_set_path = dataBasePath #+ data_set_name + '/'
    # df, dft = do_single_data_set(data_set_path)
    # dataBasePath = data_set_path

    df, dft = do_all_data_sets(data_set_path)
    data_set_path = dataBasePath + data_set_name + '/'
    

    summaryInfo(df, 'Final Cumulative')
    summaryInfo(dft, 'Final Cumulative Tournament')

    df = dedupe(df)
    dft = dedupe(dft)
    summaryInfo(df, 'Final Cumulative (Deduped)')
    summaryInfo(dft, 'Final Cumulative Tournament (Deduped)')


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

    features_valid = df.values[test_count: (test_count + validation_count), 0:21]
    labels_valid = df.values[test_count: (test_count + validation_count), 21]


    features_train = df.values[(test_count + validation_count):, 0:21]
    labels_train = df.values[(test_count + validation_count):, 21]

    unsupervised_df = pd.concat([pd.DataFrame(features_train), pd.DataFrame(features_test), pd.DataFrame(tournament)])

    unsup_dataset_size = len(unsupervised_df.values)
    unsup_test_count = int(unsup_dataset_size * test_pc)

    unsup_features_test = unsupervised_df.values[0:unsup_test_count,0:21]
    unsup_features_train = unsupervised_df.values[unsup_test_count:,0:21]

    # print(features[0])
    # print(labels[0])
    # print(tournament[0])
    # print(tournament_ids[0])
    # print(unsup_features_train[0])
    # print(unsupervised_df)

    print('features:', np.shape(features))
    print('labels:',np.shape(labels))

    print('features-train:', np.shape(features_train))
    print('labels-train:',np.shape(labels_train))

    print('features-test:', np.shape(features_test))
    print('labels-test:',np.shape(labels_test))

    print('features-valid:', np.shape(features_valid))
    print('labels-valid:',np.shape(labels_valid))

    print('unsup_features_train:', np.shape(unsup_features_train))
    print('unsup_features_test:', np.shape(unsup_features_test))
    
    print('tournament:', np.shape(tournament))
    print('tournament_ids:',np.shape(tournament_ids))

    features = np.reshape(features, [-1, 21, 1])
    tournament = np.reshape(tournament, [-1, 21, 1])

    print(np.shape(features))

    np.save(data_set_path + 'tournament-' + data_set_name + '.npy', tournament)
    np.save(data_set_path + 'tournament_ids-' + data_set_name + '.npy', tournament_ids)

    np.save(data_set_path + 'features-' + data_set_name + '.npy', features)
    np.save(data_set_path + 'labels-' + data_set_name + '.npy', labels)

    np.save(data_set_path + 'features-train-' + data_set_name + '.npy', features_train)
    np.save(data_set_path + 'labels-train-' + data_set_name + '.npy', labels_train)

    np.save(data_set_path + 'features-valid-' + data_set_name + '.npy', features_valid)
    np.save(data_set_path + 'labels-valid-' + data_set_name + '.npy', labels_valid)
    
    np.save(data_set_path + 'features-test-' + data_set_name + '.npy', features_test)
    np.save(data_set_path + 'labels-test-' + data_set_name + '.npy', labels_test)

    np.save(data_set_path + 'unsup-features-train-' + data_set_name + '.npy', unsup_features_train)
    np.save(data_set_path + 'unsup-features-test-' + data_set_name + '.npy', unsup_features_test)

    if show_stats:
        std = np.std(features, dtype=np.float64)
        mean = np.mean(features, dtype=np.float64)
        print ('features: Std=' + str(std), 'Mean=' + str(mean))

        std = np.std(features_valid, dtype=np.float64)
        mean = np.mean(features_valid, dtype=np.float64)
        print ('features_valid: Std=' + str(std), 'Mean=' + str(mean))

        std = np.std(tournament, dtype=np.float64)
        mean = np.mean(tournament, dtype=np.float64)
        print ('tournament: Std=' + str(std), 'Mean=' + str(mean))

        std = np.std(unsup_features_test, dtype=np.float64)
        mean = np.mean(unsup_features_test, dtype=np.float64)
        print ('unsupervised test: Std=' + str(std), 'Mean=' + str(mean))

        std = np.std(unsup_features_train, dtype=np.float64)
        mean = np.mean(unsup_features_train, dtype=np.float64)
        print ('unsupervised train: Std=' + str(std), 'Mean=' + str(mean))

        std = np.std(unsupervised_df.values, dtype=np.float64)
        mean = np.mean(unsupervised_df.values, dtype=np.float64)
        print ('unsupervised train: Std=' + str(std), 'Mean=' + str(mean))


prep_datasets(True)
