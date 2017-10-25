'''
Perceptron
    activation = bias + ZIGMA( weighti × xi )
    prediction = 1.0 IF activation ≥ 0.0 ELSE 0.0

@author: jame phankosol
'''

# Importing the library
from random import seed
from data_preparation import load_csv, evaluation_algorithm, evaluation_metrics


# Make a prediction with weights
def predict(row, weights):
    '''
    weights[] is adjusted by gradient algorithm
    ** weight[0] is bias
    :param row:
    :param weights:
    :return:
    '''
    activation = weights[0]
    for i in range(0, len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch, print_epoch=False):
    '''
    w(t + 1) = w(t) + learning rate × (expected(t) − predicted(t)) × x(t)
    bias(t + 1) = bias(t) + learning rate × (expected(t) − predicted(t))
    :param train:
    :param l_rate:
    :param n_epoch:
    :return:
    '''
    weights = [0.0 for _ in range(0, len(train[0]))]  # weights[0] is bias
    for epoch in range(0, n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + l_rate * error
            for i in range(0, len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        if print_epoch:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights


# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
    weights = train_weights(train, l_rate, n_epoch)
    predictions = []
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


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

        predicted = perceptron(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        accuracy = evaluation_metrics.accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


# Test the Perceptron algorithm on the sonar dataset
seed(1)
# load and prepare data
filename = 'file_collection/sonar.all-data.csv'
dataset = load_csv.load_csv(filename)
for i in range(len(dataset[0]) - 1):
    load_csv.str_column_to_float(dataset, i)
# convert string class to integers
load_csv.str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 3
l_rate = 0.01
n_epoch = 500
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

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
# weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
# for row in dataset:
#     prediction = predict(row, weights)
#     print("Expected=%d, Predicted=%d" % (row[-1], prediction))

# # Calculate weights
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
# l_rate = 0.1
# n_epoch = 5
# weights = train_weights(dataset, l_rate, n_epoch, print_epoch=True)
# print(weights)
