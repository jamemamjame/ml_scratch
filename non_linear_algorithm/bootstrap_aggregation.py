'''
Bootstrap Aggregation
" Building multiple models from samples of your training data, called bagging, can reduce this variance "

@author: jame phankosol
'''

# Importing the library
from random import randrange, seed
from data_preparation import load_csv, data_scaling, evaluation_algorithm, evaluation_metrics
from non_linear_algorithm import decision_tree as dtree


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio=1.0):
    sample = []
    n_sample = round(len(dataset) * ratio)
    for n in range(n_sample):
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [dtree.predict(tree, row) for tree in trees]
    return max(predictions, key=predictions.count)


# Bootstrap Aggregation Algorithm
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    '''
    this function implement to random_forest
    :param train:
    :param test:
    :param max_depth:
    :param min_size:
    :param sample_size:
    :param n_trees:
    :return:
    '''
    trees = []
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = dtree.build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = evaluation_algorithm.cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = evaluation_metrics.accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# # Test bagging on the sonar dataset
# seed(1)
# # load and prepare data
# filename = '/Users/jamemamjame/Computer-Sci/machine_learning_algorithms_from_scratch/ml_from_scratch/file_collection/sonar.all-data.csv'
# dataset = load_csv.load_csv(filename)
# # convert string attributes to integers
# for i in range(len(dataset[0]) - 1):
#     load_csv.str_column_to_float(dataset, i)
# # convert class column to integers
# load_csv.str_column_to_int(dataset, len(dataset[0]) - 1)
# # evaluate algorithm
# n_folds = 5
# max_depth = 6
# min_size = 2
# sample_size = 0.50
# for n_trees in [1, 5, 10]:
#     scores = evaluate_algorithm(dataset, bagging, n_folds, max_depth, min_size, sample_size,
#                                 n_trees)
#     print('Trees: %d' % n_trees)
#     print('Scores: %s' % scores)
#     print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
