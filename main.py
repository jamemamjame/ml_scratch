from data_preparation import load_csv

filename = 'file_collection/pima-indians-diabetes.csv'
dataset = load_csv.load_csv(filename)
print('Loaded data file {} with {} rows and {} columns'.format(filename, len(dataset), len(dataset[0])))

# convert string columns to float
for i in range(len(dataset[0])):
    load_csv.str_column_to_float(dataset, i)
print(dataset[0])
# Estimate mean and standard deviation
means = load_csv.column_means(dataset)
stdevs = load_csv.column_stdevs(dataset, means)
# standardize dataset
load_csv.standardize_dataset(dataset, means, stdevs)
print(dataset[0])
