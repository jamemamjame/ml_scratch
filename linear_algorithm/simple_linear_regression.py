'''
Simple Linear Regression

@author: jame phankosol
'''

# Import the library
from data_preparation import load_csv, evaluation_algorithm
from data_preparation.evaluation_metrics import rmse_metric
from random import seed


# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))


# Calculate the variance of a list of numbers
def variance(values, mean):
    variance = .0
    for value in values:
        variance += (value - mean) ** 2
    return variance


# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = .0
    for i in range(0, len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


# Calculate coefficients
def coefficients(dataset):
    X = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    X_mean, y_mean = mean(X), mean(y)

    #  cal coefficient
    b1 = covariance(X, X_mean, y, y_mean) / float(variance(X, X_mean))
    b0 = y_mean - b1 * X_mean
    return (b0, b1)


# Simple linear regression algorithm
def simple_linear_regression(train, test):
    b0, b1 = coefficients(train)
    predictions = []
    for row in test:
        x = row[0]
        yhat = b0 + b1 * x
        predictions.append(yhat)
    return predictions


# Evaluate regression algorithm on training dataset (specific case)
def evaluate_algorithm_specialcase(dataset, algorithm):
    '''
    This is special method which created for non-split dataset to train ans test.
    Assume train = test = dataset(paremeter)
    :param dataset:
    :param algorithm:
    :return: rmse
    '''
    test_set = []
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(dataset, test_set)
    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# calculate mean and variance
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# x = [row[0] for row in dataset]
# y = [row[1] for row in dataset]
# mean_x, mean_y = mean(x), mean(y)
# var_x, var_y = variance(x, mean_x), variance(y, mean_y)
# print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
# print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
#
# # calculate covariance
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# x = [row[0] for row in dataset]
# y = [row[1] for row in dataset]
# mean_x, mean_y = mean(x), mean(y)
# covar = covariance(x, mean_x, y, mean_y)
# print('Covariance: %.3f' % (covar))
#
# # calculate coefficients
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# b0, b1 = coefficients(dataset)
# print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
# plt.scatter([row[0] for row in dataset], [row[1] for row in dataset])
# plt.grid(True)
# plt.show()

# # Test simple linear regression
# dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# rmse = evaluate_algorithm_specialcase(dataset, simple_linear_regression)
# print('RMSE: %.3f' % (rmse))


# Simple linear regression on insurance dataset
seed(1)
# load and prepare data
filename = 'file_collection/insurance.csv'
dataset = load_csv.load_csv(filename)
for i in range(len(dataset[0])):
    load_csv.str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.8
train, test = evaluation_algorithm.train_test_split(dataset, split)
ypred = simple_linear_regression(train, test)
actual = [row[1] for row in test]

