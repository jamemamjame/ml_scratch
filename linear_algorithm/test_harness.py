'''
Test Harness
We cannot know which algorithm will be best for a given problem. Therefore, we need to design a test harness that we can use to evaluate different machine learning algorithms

@author: jame phankosol
'''

# Import the library
from data_preparation.evaluation_metrics import accuracy_metric, rmse_metric
from data_preparation.evaluation_algorithm import train_test_split, cross_validation_split


# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = []

    # copy test set for protect cheating (protect lookahead answer), then we remove last index (class, answer)
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)

    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]  # extract the class (answer)

    rmse = rmse_metric(actual, predicted)
    # assume this is classification problem.
    # but this could be changed to mean squared error for regression problems.
    return rmse


# Evaluate an algorithm using a cross-validation split
def evaluate_algorithm2(dataset, algorithm, n_fold, *args):
    folds = cross_validation_split(dataset, n_fold)
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

        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# # Test the train/test harness
# seed(1)
# # load and prepare data
# filename = 'file_collection/pima-indians-diabetes.csv'
# dataset = data_preparing.load_csv(filename)
# for i in range(len(dataset[0])):
#     data_preparing.str_column_to_float(dataset, i)
# # evaluate algorithm
# split = 0.6
# accuracy = evaluate_algorithm(dataset, zero_rule_algorithm_classification, split)
# print('Accuracy: %.3f%%' % (accuracy))

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# Test cross validation test harness
# seed(1)
# # load and prepare data
# filename = 'file_collection/pima-indians-diabetes.csv'
# dataset = data_preparing.load_csv(filename)
# for i in range(len(dataset[0])):
#     data_preparing.str_column_to_float(dataset, i)
# # evaluate algorithm
# n_folds = 5
# scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, n_folds)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores) / len(scores)))
