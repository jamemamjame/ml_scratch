'''
Multivariate Linear Regression
    y = b0 + b1 × x1 + b2 × x2 + ...
Stochastic Gradient Descent
    b = b − learning rate × error × x

@author: jame phankosol
page 64
'''
# Importing the library
from random import seed

from data_preparation import load_csv, data_scaling


# Make a prediction with coefficients
def predict(row, coefficients):
    '''
    predicts an output value for a row given a set of coefficients.
        y = b0 + b1 × x1 + b2 × x2 + ...
    :param row: vector or feture or attribute
    :param coefficients: list of coefficient
    :return:
    '''
    yhat = coefficients[0]  # it's mean b0
    for i in range(0, len(row) - 1):  # don't calculate with last index (answer)
        yhat += row[i] * coefficients[i + 1]
    return yhat


# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch=50, print_epoch=False):
    '''
    error = prediction − expected
    b1(t + 1) = b1(t) − learning rate × error(t) × x1(t)    ถ้า error(-) แปลว่าเราทำนายได้ค่าต่ำกว่าของจริง จึงมีการทำให้ค่า b เกิดลบลบได้บวก ความชันกราฟสูงขึ้น
    b0(t + 1) = b0(t) − learning rate × error(t)

    We can set constant_error which use for compare that if error < constant_error we will break.
    :param train:
    :param l_rate:
    :param n_epoch:
    :return:
    '''
    coef = [0.0 for i in range(len(train[0]))]
    # loop of n_epoch
    for epoch in range(0, n_epoch):
        sum_square_error = 0.0

        # loop for predict yhat from each vector while update coefficient in same time
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_square_error += error ** 2

            # change value of parameter coef[]
            coef[0] = coef[0] - l_rate * error  # b0 = b0 - learning_rate * error

            # loop for update coefficient which dot cal for b0
            for i in range(0, len(coef) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]  # b1 = b1 - learning_rate * error * x5
        if print_epoch:
            print('>epoch=%d, lrate=%.3f, sum_square_error=%.3f' % (epoch, l_rate, sum_square_error))
    return coef


# Evaluate an algorithm using a cross-validation split
def evaluate_algorithm(dataset, algorithm, n_fold, *args):
    from data_preparation.evaluation_algorithm import cross_validation_split
    from data_preparation.evaluation_metrics import rmse_metric
    folds = cross_validation_split(dataset, n_fold)
    scores = []
    # loop for remove fold from folds
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])  # union a set of vector

        test_set = []
        # copy test set for protect cheating (protect lookahead answer), then we remove last index (class, answer)
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores


# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
    coef = coefficients_sgd(train, l_rate, n_epoch)
    predictions = []
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return predictions


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


# Linear Regression on wine quality dataset
seed(1)
# load and prepare data
filename = '/Users/jamemamjame/Computer-Sci/machine_learning_algorithms_from_scratch/ml_from_scratch/file_collection/winequality-white.csv'
dataset = load_csv.load_csv(filename)
for i in range(len(dataset[0])):
    load_csv.str_column_to_float(dataset, i)
# normalize
minmax = data_scaling.dataset_minmax(dataset)
data_scaling.normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.01
n_epoch = 50
scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores) / float(len(scores))))
