'''
Logistic Regression:
    A key difference from linear regression is that the output value being modeled is a binary value (0 or 1) rather than a numeric value.
            yhat = ________ 1.0 ___________
                              −(b0+b1×x1)
                       1.0 + e

@author: jame phankosol
'''

# Importing the library
import math
from random import seed
from data_preparation import load_csv, data_scaling, evaluation_algorithm, evaluation_metrics


# Make a prediction with coefficients
def predict(row, coefficients):
    '''
            yhat = ________ 1.0 ___________
                              −(b0+b1×x1)
                       1.0 + e
    :param row:
    :param coefficients:
    :return: class (float, must round to 0 or 1 by self)
    '''
    yhat = coefficients[0]
    for i in range(0, len(row) - 1):  # don't cal the last index
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + math.exp(-yhat))


# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch=50, print_epoch=False):
    '''
    b1(t + 1) = b1(t) + learning rate × (y(t) − yhat(t)) × yhat(t) × (1 − yhat(t)) × x1(t)  ; error = (y(t) − yhat(t))
    b0(t + 1) = b0(t) + learning rate × (y(t) − yhat(t)) × yhat(t) × (1 − yhat(t))
    :param train:
    :param l_rate:
    :param n_epoch:
    :return:
    '''
    coef = [0.0 for _ in range(0, len(train[0]))]
    for epoch in range(0, n_epoch):
        sum_error = 0.0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            sum_error += error ** 2
            coef[0] = coef[0] + l_rate * error * yhat * (1 - yhat)
            for i in range(0, len(coef) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1 - yhat) * row[i]
        if print_epoch:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef


# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    coefficients = coefficients_sgd(train, l_rate, n_epoch)
    prediction = []
    for row in test:
        yhat = predict(row, coefficients)
        yhat = round(yhat)
        prediction.append(yhat)
    return prediction


# Evaluate an algorithm using a cross-validation split
def evaluate_algorithm(dataset, algorithm, n_fold, *args):
    folds = evaluation_algorithm.cross_validation_split(dataset, n_fold)
    scores = []
    # loop for remove fold from folds
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])  # union the mini list of vector

        test_set = []
        # copy test set for protect cheating (protect lookahead answer), then we remove last index (class, answer)
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        accuracy = evaluation_metrics.accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# # test predictions
# dataset = [[2.7810836, 2.550537003, 0],
#            [1.465489372, 2.362125076, 0],
#            [3.396561688, 4.400293529, 0],
#            [1.38807019, 1.850220317, 0],
#            [3.06407232, 3.005305973, 0],
#            [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1],
#            [6.922596716, 1.77106367, 1],
#            [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]]
# coef = [-0.406605464, 0.852573316, -1.104746259]
# for row in dataset:
#     yhat = predict(row, coef)
#     print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))


# # Calculate coefficients
# dataset = [[2.7810836, 2.550537003, 0],
#            [1.465489372, 2.362125076, 0],
#            [3.396561688, 4.400293529, 0],
#            [1.38807019, 1.850220317, 0],
#            [3.06407232, 3.005305973, 0],
#            [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1],
#            [6.922596716, 1.77106367, 1],
#            [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]]
# l_rate = 0.3
# n_epoch = 100
# coef = coefficients_sgd(dataset, l_rate, n_epoch, print_epoch=True)
# print(coef)


# # Test the logistic regression algorithm on the diabetes dataset
# seed(1)
# # load and prepare data
# filename = 'file_collection/pima-indians-diabetes.csv'
# dataset = load_csv.load_csv(filename)
# for i in range(len(dataset[0])):
#     load_csv.str_column_to_float(dataset, i)
# minmax = data_scaling.dataset_minmax(dataset)
# data_scaling.normalize_dataset(dataset, minmax)
# # evaluate algorithm
# n_folds = 5
# l_rate = 0.1
# n_epoch = 100
# scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

# Test LVQ on Ionosphere dataset
seed(1)
# load and prepare data
filename = 'file_collection/ionosphere.csv'
dataset = load_csv.load_csv(filename)
for i in range(len(dataset[0]) - 1):
    load_csv.str_column_to_float(dataset, i)
# convert class column to integers
load_csv.str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
