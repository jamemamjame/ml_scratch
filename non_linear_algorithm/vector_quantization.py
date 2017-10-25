'''
Vector Quantization

@author: jame phankosol
'''

# Importing the library
import math
from random import randrange
from random import seed
from data_preparation import evaluation_algorithm, evaluation_metrics, load_csv


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(0, len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
    '''
    ดูว่า codebook ตัวไหนใน codebooks ที่ใกล้เคียงกับ test_row ที่สุด
    :param codebooks:
    :param test_row:
    :return:
    '''
    max_dist, best_codebook = float('inf'), None
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        if dist < max_dist:
            max_dist, best_codebook = dist, codebook
    return best_codebook


# Create a random codebook vector
def random_codebook(train):
    '''
    สุ่ม vector จากค่าใน training_data ที่มี
    :param train:
    :return:
    '''
    n_records = len(train)
    n_fetures = len(train[0])
    codebook = [train[randrange(0, n_records)][i] for i in range(0, n_fetures)]
    return codebook


# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs, print_epoch_error=False):
    codebooks = [random_codebook(train) for _ in range(0, n_codebooks)]
    for epoch in range(0, epochs):
        # the learning rate is adjusted so that it has maximum effect in the first epoch and less effect as training
        # continues until it has a minimal effect in the final epoch
        sum_error = 0.0
        rate = lrate * (1 - (epoch / float(epochs)))
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            for i in range(0, len(row) - 1):
                error = row[i] - bmu[i]
                sum_error += error ** 2
                if bmu[-1] == row[-1]:  # if same class
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
        if print_epoch_error:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
    return codebooks


# Make a prediction with codebook vectors
def predict(codebooks, test_row):
    bmu = get_best_matching_unit(codebooks, test_row)
    return bmu[-1]


# LVQ Algorithm
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    return (predictions)


# Evaluate an algorithm using a cross validation split
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
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Test best matching unit function
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
# test_row = dataset[0]
# bmu = get_best_matching_unit(dataset, test_row)
# print(bmu)


# # Test the training function
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
# learn_rate = 0.3
# n_epochs = 10
# n_codebooks = 2
# codebooks = train_codebooks(dataset, n_codebooks, learn_rate, n_epochs, print_epoch_error=True)
# print('Codebooks: %s' % codebooks)


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
learn_rate = 0.3
n_epochs = 50
n_codebooks = 20
scores = evaluate_algorithm(dataset, learning_vector_quantization, n_folds, n_codebooks,learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
