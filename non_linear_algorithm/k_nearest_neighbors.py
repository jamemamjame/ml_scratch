'''
k-Nearest Neighbors

@author: jame phankosol
'''

# Importing the library
import math
from random import seed
from data_preparation import evaluation_algorithm, evaluation_metrics, load_csv


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(0, len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        distance = euclidean_distance(test_row, train_row)
        distances.append((train_row, distance))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors=5):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(output_values, key=output_values.count)
    return prediction


# Make a prediction with neighbors
def predict_regression(train, test_row, num_neighbors=5):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = sum(output_values) / float(num_neighbors)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = []
    for test_row in test:
        prediction = predict_classification(train, test_row, num_neighbors)
        predictions.append(prediction)
    return predictions


# Classification
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = evaluation_algorithm.cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])

        test_set = []
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        accuracy = evaluation_metrics.accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Regression
def evaluate_algorithm2(dataset, algorithm, n_folds, *args):
    folds = evaluation_algorithm.cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])

        test_set = []
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        rmse = evaluation_metrics.rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores


#
#
#
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# Test the kNN on the Abalone dataset
seed(1)
# load and prepare data
filename = 'file_collection/abalone.csv'
dataset = load_csv.load_csv(filename)
for i in range(1, len(dataset[0])):
    load_csv.str_column_to_float(dataset, i)
# convert first column to integers
load_csv.str_column_to_int(dataset, 0)
# evaluate algorithm
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

# Test the kNN on the Abalone dataset
seed(1)
# load and prepare data
filename = 'file_collection/abalone.csv'
dataset = load_csv.load_csv(filename)
for i in range(1, len(dataset[0])):
    load_csv.str_column_to_float(dataset, i)
# convert first column to integers
load_csv.str_column_to_int(dataset, 0)
# evaluate algorithm
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores) / float(len(scores))))
