'''
Data Scaling

@author: jame phankosol
'''
import numpy as np

# Find the min and max values for each column
def dataset_minmax(dataset):
    '''
    retuen list of tuple (min, max) for each column
    :param dataset:
    :return:
    '''
    minmax = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append((value_min, value_max))
    return minmax


# Rescale dataset columns to the any range [desired_min, desired_max]
def normalize_dataset(dataset, minmax, desired_min=-1, desired_max=1):
    '''
    rescale data's value
    :param dataset:
    :param minmax:
    :param desired_min:
    :param desired_max:
    :return:
    '''
    desired_range = desired_max - desired_min
    for i in range(0, len(dataset[0])):
        orginal_min, original_max = minmax[i][0], minmax[i][1]
        orginal_range = original_max - orginal_min
        for row in dataset:
            row[i] = (desired_range * ((row[i] - orginal_min) / orginal_range)) + desired_min


# calculate column means
def column_means(dataset):
    means = []
    for i in range(0, len(dataset[0])):
        sum = 0
        for row in dataset:
            sum += row[i]
        mean = sum / float(len(dataset))
        means.append(mean)
    return means


# calculate column standard deviations (SD)
def column_stdevs(dataset, means):
    stdevs = []
    for i in range(0, len(dataset[0])):
        variance = 0
        for row in dataset:
            variance += (row[i] - means[i]) ** 2
        stdevs.append(np.sqrt(variance / float(len(dataset) - 1)))
    return stdevs


# standardize dataset
def standardize_dataset(dataset, means, stdevs):
    '''
    converting data to center
    :param dataset:
    :param means:
    :param stdevs:
    :return:
    '''
    for row in dataset:
        for i in range(0, len(dataset[0])):
            row[i] = (row[i] - means[i]) / stdevs[i]
