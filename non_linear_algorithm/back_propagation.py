'''
Back-Propagation

@author: jame phankosol
'''

# Importing the library
from random import random, seed
import math
from data_preparation import evaluation_algorithm, evaluation_metrics, load_csv, data_scaling


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    '''
    create network -> [hidden_layer[{dict}], output_layer[{dict}]]
    :param n_inputs:
    :param n_hidden:
    :param n_outputs:
    :return:
    '''
    network = []

    # hidden layer collects weight + bias
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    network.append(hidden_layer)

    # output layer collects weight + bias
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)

    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    '''
    activation = bias + ZIGMA(weighti × inputi)
    :param weights:
    :param inputs:
    :return:
    '''
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    '''
    sigmoid, tanh (hyperbolic tangent), rectifier
    all of this can use for transfer
    in this tutorial, we use sigmoid function

    transfer คือการปรับค่า activation ให้อยู่ในรูปแบบใหม่ ผ่านการใช้ฟังก์ชั่นตัวอย่างข้างต้น ในที่นี้ใช้ sigmoid
    :param activation:
    :return:
    '''
    return 1 / (1 + math.exp(-activation))


# Forward-propagate input to a network output
def forward_propagate(network, row):
    '''
    All of the outputs from one layer become inputs to the neurons on the next layer.
    :param network:
    :param row:
    :return:
    '''
    inputs = row
    for layer in network:
        outputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            outputs.append(neuron['output'])
        inputs = outputs
    # last time, when finish loop -> output(class_value) = current input = last outputs
    return outputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    '''
    This is a slope equation which diff' from transfer function

    ปล. สมการในข้อนี้ ดิฟมาจากสมการ sigmoid
    :param output:
    :return:
    '''
    # (use with sigmoid)
    return output * (1.0 - output)


# Back-propagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        # if current_layer is output_layer
        if i == len(network) - 1:
            for j, neuron in enumerate(network[i]):  # network[i] is current_layer
                '''
                    error = (expected − output) × transfer derivative(output)
                '''
                error = (expected[j] - neuron['output']) * transfer_derivative(neuron['output'])
                neuron['delta'] = error

        # if current_layer is input_layer or hidden_layer
        else:
            for j, neuron in enumerate(network[i]):
                error = 0.0
                for neuron_2 in network[i + 1]:  # network[i + 1] is a layer on right hand (close to the output_layer)
                    '''
                        error = (weightk × errorj ) × transfer derivative(output)
                    '''
                    error += (neuron_2['weights'][j] * neuron_2['delta']) * transfer_derivative(neuron['output'])
                neuron['delta'] = error


# Update network weights with error
def update_weights(network, inputs, l_rate):
    cur_inputs = inputs[:-1]
    for i, layer in enumerate(network):
        # if not is first hidden_layer
        if i != 0:
            cur_inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in layer:
            for j, input in enumerate(cur_inputs):
                neuron['weights'][j] += l_rate * neuron['delta'] * input
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, print_epoch=False):
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            output = forward_propagate(network, row)
            expected = [0 for _ in range(n_outputs)]
            expected[row[-1]] = 1
            for i in range(n_outputs):
                sum_error += (expected[i] - output[i]) ** 2
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        if print_epoch:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = []
    for row in test:
        prediction = predict(network, row)
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


#
#
#
# --- Test --- Test --- Test --- Test --- Test --- Test --- Test --- Test --- Test --- Test --- Test ---
#
#
# # Test initializing a network
# seed(1)
# network = initialize_network(2, 1, 2)
# for layer in network:
#     print(layer)


# # test forward propagation
# network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#            [{'weights': [0.2550690257394217, 0.49543508709194095]},
#             {'weights': [0.4494910647887381, 0.651592972722763]}]]
# row = [1, 0, None]
# output = forward_propagate(network, row)
# print(output)


# # test backpropagation of error
# network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#            [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
#             {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
# expected = [0, 1]
# backward_propagate_error(network, expected)
# for layer in network:
#     print(layer)


# # Test training backprop algorithm
# seed(1)
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
# n_inputs = len(dataset[0]) - 1
# n_outputs = len(set([row[-1] for row in dataset]))
# network = initialize_network(n_inputs, 2, n_outputs)
# train_network(network, dataset, 0.5, 20, n_outputs, print_epoch=True)
# for layer in network:
#     print(layer)



# # Test Backprop on Seeds dataset
# seed(1)
# # load and prepare data
# filename = '/Users/jamemamjame/Computer-Sci/machine_learning_algorithms_from_scratch/ml_from_scratch/file_collection/seeds_dataset.csv'
# dataset = load_csv.load_csv(filename)
# for i in range(len(dataset[0])):
#     load_csv.str_column_to_float(dataset, i)
# # normalize input variables
# minmax = data_scaling.dataset_minmax(dataset)
# data_scaling.normalize_dataset(dataset, minmax)
# # convert class column to integers
# load_csv.str_column_to_int(dataset, len(dataset[0]) - 1)
# # evaluate algorithm
# n_folds = 5
# l_rate = 0.3
# n_epoch = 500
# n_hidden = 5
# scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# Test Backprop on Ionosphere dataset
seed(1)
# load and prepare data
filename = '/Users/jamemamjame/Computer-Sci/machine_learning_algorithms_from_scratch/ml_from_scratch/file_collection/ionosphere.csv'
dataset = load_csv.load_csv(filename)
for i in range(len(dataset[0]) - 1):
    load_csv.str_column_to_float(dataset, i)
# convert class column to integers
load_csv.str_column_to_int(dataset, len(dataset[0]) - 1)
# normalize input variables
minmax = data_scaling.dataset_minmax(dataset)
data_scaling.normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
