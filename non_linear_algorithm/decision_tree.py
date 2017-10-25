'''
Classification and Regression Tree (CART)

@author: jame phankosol
'''

# Importing the library
import math
from random import seed
from data_preparation import load_csv, data_scaling, evaluation_algorithm, evaluation_metrics


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


# Select the best split point for a dataset
def get_split(dataset, print_sub_gini=False):
    '''
    this function given a dataset and process it to create a node that contain left, right, None child
    :param dataset:
    :param print_sub_gini:
    :return: node
    '''
    class_values = list(set([row[-1] for row in dataset]))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(0, len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index=index, value=row[index], dataset=dataset)
            gini_score = gini_index(groups=groups, class_values=class_values)
            if print_sub_gini:
                print('X%d < %.3f, Gini = %.3f' % (index + 1, row[index], gini_score))
            if gini_score < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini_score, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


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


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    '''
    :param train:
    :param test:
    :param max_depth: maximum depth of tree that possible
    :param min_size: minimum size of data in each node that possible
    :return:
    '''
    tree = build_tree(train, max_depth, min_size)
    predictions = []
    for row in test:
        prediction = predict(tree, row)
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

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        accuracy = evaluation_metrics.accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Test get_split
# dataset = [[2.771244718, 1.784783929, 0],
#            [1.728571309, 1.169761413, 0],
#            [3.678319846, 2.81281357, 0],
#            [3.961043357, 2.61995032, 0],
#            [2.999208922, 2.209014212, 0],
#            [7.497545867, 3.162953546, 1],
#            [9.00220326, 3.339047188, 1],
#            [7.444542326, 0.476683375, 1],
#            [10.12493903, 3.234550982, 1],
#            [6.642287351, 3.319983761, 1]]
# split = get_split(dataset, print_sub_gini=True)
# print('Split: [X%d < %.3f]' % ((split['index']), split['value']))


# Test build_tree
# dataset = [[2.771244718, 1.784783929, 0],
#            [1.728571309, 1.169761413, 0],
#            [3.678319846, 2.81281357, 0],
#            [3.961043357, 2.61995032, 0],
#            [2.999208922, 2.209014212, 0],
#            [7.497545867, 3.162953546, 1],
#            [9.00220326, 3.339047188, 1],
#            [7.444542326, 0.476683375, 1],
#            [10.12493903, 3.234550982, 1],
#            [6.642287351, 3.319983761, 1]]
# tree = build_tree(dataset, 1, 1)
# print_tree(tree)

# Test predict
# contrived dataset
# dataset = [[2.771244718, 1.784783929, 0],
#            [1.728571309, 1.169761413, 0],
#            [3.678319846, 2.81281357, 0],
#            [3.961043357, 2.61995032, 0],
#            [2.999208922, 2.209014212, 0],
#            [7.497545867, 3.162953546, 1],
#            [9.00220326, 3.339047188, 1],
#            [7.444542326, 0.476683375, 1],
#            [10.12493903, 3.234550982, 1],
#            [6.642287351, 3.319983761, 1]]
# # predict with a stump
# stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
# for row in dataset:
#     prediction = predict(stump, row)
#     print('Expected=%d, Got=%d' % (row[-1], prediction))


# # Test CART on Bank Note dataset
# seed(1)
# # load and prepare data
# filename = 'file_collection/data_banknote_authentication.csv'
# dataset = load_csv.load_csv(filename)
# # convert string attributes to integers
# for i in range(len(dataset[0])):
#     load_csv.str_column_to_float(dataset, i)
# # evaluate algorithm
# n_folds = 5
# max_depth = 5
# min_size = 10
# scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
