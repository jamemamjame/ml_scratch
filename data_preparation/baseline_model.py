'''
@author: jame phankosol

Baseline Models
    Random Prediction Algorithm.
    Zero Rule Algorithm.    ***
ใช้เพื่อเปรียบเทียบกับที่เรา predict เอง ว่าที่เราทำเองอย่างน้อยต้องชนะ Baseline Models พวกนี้นะ
คำว่าชนะในที่นี้หมายถึง model ที่เราสร้างมา predict ต้องมีความแม่นยำที่สูงกว่า ไม่ว่าจะวัดด้วย F1, Recall, Accuracy, RMSE, MAE, etc.
'''

# Import the library
import random


# Generate random predictions
def random_algorithm(train, test):
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = []
    for i in range(0, len(test)):
        index = random.randrange(0, len(unique))
        predicted.append(unique[index])
    return predicted


# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(output_values, key=output_values.count)
    predicted = [prediction for i in range(0, len(test))]
    return predicted


# zero rule algorithm for regression
def zero_rule_algorithm_regression(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(output_values) / float(len(output_values)) # this use mean, also can use median/mode
    predicted = [prediction for i in range(0, len(test))]
    return predicted

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# random.seed(1)
# train = [[0], [1], [0], [1], [0], [1]]
# test = [[None], [None], [None], [None]]
# predictions = random_algorithm(train, test)
# print(predictions)
#
# train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
# test = [[None], [None], [None], [None]]
# predictions = zero_rule_algorithm_classification(train, test)
# print(predictions)
#
# train = [[10], [15], [12], [15], [18], [20]]
# test = [[None], [None], [None], [None]]
# predictions = zero_rule_algorithm_regression(train, test)
# print(predictions)