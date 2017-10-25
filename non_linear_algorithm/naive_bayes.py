'''
Naive Bayes
extension: normal_distribution https://en.wikipedia.org/wiki/Normal_distribution

@author: Jame Phankosol
'''

# Importing the library
import math
from data_preparation import evaluation_algorithm, evaluation_metrics, load_csv


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    '''
    ทำการแยก dataset ว่าคลาสนี้มี vector ไหนบ้าง
    :param dataset:
    :return:
    '''
    # create dict()
    sepatated = {}
    for vector in dataset:
        class_value = vector[-1]
        if not class_value in sepatated:
            sepatated[class_value] = []
        sepatated[class_value].append(vector)
    return sepatated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    varince = 0.0
    for x in numbers:
        varince += (x - avg) ** 2
    return math.sqrt(varince / (float(len(numbers)) - 1))


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del summaries[-1]
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    '''

    :param dataset:
    :return: (means, stdev, length) for each column
    '''
    separate = separate_by_class(dataset)
    summarize = {}
    for class_value, rows in separate.items():
        summarize[class_value] = summarize_dataset(rows)
    return summarize


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    '''
    normal distribution calculate
    :param x:
    :param mean:
    :param stdev:
    :return:
    '''
    exponent = math.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    '''
    จาก     P(class|data) = ______P(X|class) × P(class)______
                                        P(X)
    ลดรูปเป็น:
            P(class|data) = P(X|class) × P(class)
    // ไม่ต้องหารด้วย P(data) แล้ว เพราะทุกตัวมี P(data)เท่ากัน จะเปรียบเทียบหา max_value ไม่จำเป็นต้องใช้ constant มาพิจารณา

    EX: X = [X1, X2]
        P(class = 0|X1, X2) = P(X1|class = 0) × P(X2|class = 0) × P(class = 0)

    :param summaries:
    :param row:
    :return:
    '''
    total_rows = 0
    for _, class_summaries in summaries.items():
        total_rows += class_summaries[0][2]  # get number of row from each class

    probabilities = {}
    for class_value, class_summaries in summaries.items():
        # calculate P(classA)
        # P(classA) = (number_of_row_from_classA) / (number_of_total_row)
        probabilities[class_value] = class_summaries[0][2] / float(total_rows)

        # calculate each P(X1|class = A)
        for numCol in range(0, len(class_summaries)):
            mean, stdev, count = class_summaries[numCol]
            # calculate on each attribute of row
            probabilities[class_value] *= calculate_probability(x=row[numCol], mean=mean, stdev=stdev)
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_class, best_prob = None, -1
    for (class_value, class_prob) in probabilities.items():
        if class_prob > best_prob or best_class is None:
            best_class, best_prob = class_value, class_prob
    return best_class


# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = []
    for row in test:
        predictions.append(predict(summarize, row))
    return predictions


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

        predicted = algorithm(train_set, test_set)
        actual = [row[-1] for row in fold]

        accuracy = evaluation_metrics.accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


#
#
#
#
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Test calculating class probabilities
# dataset = [[3.393533211, 2.331273381, 0],
#            [3.110073483, 1.781539638, 0],
#            [1.343808831, 3.368360954, 0],
#            [3.582294042, 4.67917911, 0],
#            [2.280362439, 2.866990263, 0],
#            [7.423436942, 4.696522875, 1],
#            [5.745051997, 3.533989803, 1],
#            [9.172168622, 2.511101045, 1],
#            [7.792783481, 3.424088941, 1],
#            [7.939820817, 0.791637231, 1]]
# summaries = summarize_by_class(dataset)
# probabilities = calculate_class_probabilities(summaries, dataset[0])
# print(probabilities)




# Test Naive Bayes on Iris Dataset
filename = 'file_collection/iris.csv'
dataset = load_csv.load_csv(filename)
for i in range(0, len(dataset[0]) - 1):
    load_csv.str_column_to_float(dataset=dataset, column=i)
load_csv.str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % (scores))
print('Means Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
