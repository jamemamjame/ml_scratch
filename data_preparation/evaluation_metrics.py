'''
@author: jame phankosol
Implement machine learning without library

Evaluation Metrics
1.  Classification Accuracy.
2.  Confusion Matrix.
3.  Mean Absolute Error.
4.  Root Mean Squared Error.
'''

# Import the library
import math


# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.0


# calculate a confusion matrix
def confusion_matrix(actual, predicted):
    '''
    actual -> horizontal, predict -> vertical
    :param actual:
    :param predicted:
    :return:
    '''
    unique = set(actual)
    n_unique = len(unique)
    matrix = [[0] * n_unique for i in range(n_unique)]

    # create dict to remember class + index
    lookup = {}
    for i, value in enumerate(unique):
        lookup[value] = i

    # generate confusion_matrix
    for i in range(0, len(actual)):
        real = lookup[actual[i]]
        pred = lookup[predicted[i]]
        matrix[real][pred] += 1
    return unique, matrix


# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
    print('(A \ P ) ' + ' '.join([str(x) for x in unique]))
    for i, val in enumerate(unique):
        print('{}      | '.format(val) + ' '.join([str(x) for x in matrix[i]]))


# Calculate mean absolute error
def mae_metric(actual, predicted):
    sum_error = .0
    for i in range(len(actual)):
        sum_error += math.fabs(actual[i] - predicted[i])
    return sum_error / float(len(actual))


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = .0
    for i in range(len(actual)):
        sum_error += (actual[i] - predicted[i]) ** 2
    return math.sqrt(sum_error / float(len(actual)))




# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# # Test accuracy
# actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# predicted = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
# accuracy = accuracy_metric(actual, predicted)
# print(accuracy)
#
# # Test confusion matrix with integers
# actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# predicted = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
# unique, matrix = confusion_matrix(actual, predicted)
# print_confusion_matrix(unique, matrix)
#
# # Test RMSE
# actual = [0.1, 0.2, 0.3, 0.4, 0.5]
# predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
# mae = mae_metric(actual, predicted)
# print(mae)
#
# # Test RMSE
# actual = [0.1, 0.2, 0.3, 0.4, 0.5]
# predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
# rmse = rmse_metric(actual, predicted)
# print(rmse)


