'''
Random Forest

@author: jame phankosol
'''

# Importing the library
import math
from random import seed, choice
from data_preparation import load_csv, data_scaling, evaluation_algorithm, evaluation_metrics
from non_linear_algorithm import bootstrap_aggregation as bagging
from math import sqrt
import non_linear_algorithm.bootstrap_aggregation as bagging


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    '''
    proportion = ______ count(class value) _________
                           count(rows)

    gini index = ZIGMA(proportioni × (1.0 − proportioni))   ; ZIGMA(i to n(numberOfClass)
    ค่า gini จะบอกได้ว่า group in groups นั้นถูกแบ่งแยกออกจากกันได้ดีแค่ไหน (ยิ่ง gini น้อย ยิ่งดี)
    จะเห็นว่าค่า gini จะน้อยได้ ถ้าเวลาคิด proportion ได้ 0 or 1
    เพราะ gini_index หาจากผลรวม proportion *(1 - proportion)   ; proportion = 0,1 เท่านั้นถึงจะได้ค่า 0 ไปบวก
    :param groups: list of integer
    :param class_values: list of integer
    :return:
    '''
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            if len(group) == 0:
                continue
            # Other solution
            # proportion = [row[-1] for row in group].count(class_value) / float(len(group))    # use list, but must use much memory

            count = 0
            for row in group:
                if row[-1] == class_value:
                    count += 1
            proportion = count / float(len(group))

            gini += proportion * (1.0 - proportion)
    return gini


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Create a terminal node value
def to_terminal(group):
    '''
    use with terminal node for predict that this is which class?
    :param group:
    :return:
    '''
    outcomes = [row[-1] for row in group]
    return max(outcomes, key=outcomes.count)


# # # Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    '''

    :param node:
    :param max_depth: This is the maximum number of nodes from the root node of the tree.
    :param min_size: This is the minimum number of training patterns that a given node is responsible for
    :param depth:
    :return:
    '''
    left, right = node['groups']
    del (node['groups'])

    # check for a no split
    if not left or not right:  # no case that left and right are [], because a terminal node will not recursive called
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # check for over deep (max depth), but still has left child & right child
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # below this, we have 2 child and not too deep, so we can recursive call to build a tree

    # process left & right child
    # left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(dataset=left)
        split(node=node['left'], max_depth=max_depth, min_size=min_size, depth=depth + 1)
    # right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(dataset=right)
        split(node=node['right'], max_depth=max_depth, min_size=min_size, depth=depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(node=root, max_depth=max_depth, min_size=min_size, depth=1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:  # instance is Integer(class)
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Print a decision tree
def print_tree(node, depth=0):
    space = '  '
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (depth * space, (node['index'] + 1), node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * space, node)))


# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set([row[-1] for row in dataset]))
    b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
    fetures = []
    fetures_choice = [i for i in range(len(dataset[0]))]
    while len(fetures) < n_features:
        tmp_feture = choice(fetures_choice)  # random
        fetures.append(tmp_feture)
        fetures_choice.remove(tmp_feture)
    for index in fetures:
        for row in dataset:
            value = row[index]
            group = test_split(index, value, dataset)
            gini_score = gini_index(group, class_values)
            if gini_score < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini_score, group
    return {'index': b_index, 'value': b_value, 'group': b_groups}


# ---------------------- above is decision tree -------------------------------------------------


def random_forest(train, test, max_depth, min_size, sample_size, n_tree, n_fetures):
    trees = []
    for i in range(n_tree):
        sample = bagging.subsample(dataset=train, ratio=sample_size)
        tree = build_tree(train=sample, max_depth=max_depth, min_size=min_size)
        trees.append(tree)


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


# Test the random forest algorithm on sonar dataset
seed(1)
# load and prepare data
filename = 'file_collection/sonar.all-data.csv'
dataset = load_csv.load_csv(filename)
# convert string attributes to integers
for i in range(0, len(dataset[0]) - 1):
    load_csv.str_column_to_float(dataset, i)  # convert class column to integers
load_csv.str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0]) - 1))
for n_trees in [6]:
    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size,
                                sample_size, n_trees, n_features)
print('Trees: %d' % n_trees)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
