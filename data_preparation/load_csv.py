'''
Load Data with CSV

@author: jame phankosol 22
'''

# Importing the standard library
from csv import reader
import numpy as np


# Load a CSV file
def load_csv(filename):
    dataset = []
    with open(filename) as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert String to Float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert String to Integer (~OneHotEncoder)
def str_column_to_int(dataset, column):
    '''
    convert string from dataset[column] to unique integer
    like THA ENG USA USA THA => 0, 1, 2, 2, 0
    :param dataset:
    :param column:
    :return:
    '''
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {}  # dict
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
