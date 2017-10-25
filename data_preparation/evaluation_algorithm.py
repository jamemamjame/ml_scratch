'''
Train Test Split, Cross Validation

@author: jame phankosol
Implement machine learning without library
'''

# Importing the standard library
import random as rd

# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
    train_data = []
    test_data = list(dataset)  # copy it, after that we will remove piece by piece
    train_data_size = split * len(dataset)

    while len(train_data) < train_data_size:
        index = rd.randrange(0, len(test_data))
        train_data.append(test_data.pop(index))
    return train_data, test_data


# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    '''
    This is K-fold algorithm, that work for small size dataset. if large then use time so much.
    :param dataset:
    :param split:
    :return:
    '''
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(0, folds):
        fold = []
        while len(fold) < fold_size:
            index = rd.randrange(0, len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

####################################################################################################################

# dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# folds = cross_validation_split(dataset, 4)
# print(folds)