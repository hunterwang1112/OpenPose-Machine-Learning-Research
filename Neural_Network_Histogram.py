import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# load a single file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None)
    return dataframe.values


# load a dataset group, such as train or test, return them in the form of DataFrame
def load_dataset_group(data_type, group, prefix, filenames):
    # create a list for x
    x = []
    # load input data
    for name in filenames:
        data = load_file(prefix + data_type + '/' + group + name + '_' + data_type + '.csv')
        x.extend(data)
    df_x = pd.DataFrame(x)
    if data_type == 'Processed_PatientSpace':
        df_x.drop(df_x.columns[0:36], axis=1, inplace=True)
    # create a list for y
    y = []
    # load output data
    for name in filenames:
        data = load_file(prefix + 'Processed_Label/' + group + name + '_label.csv')
        y.extend(data)
    df_y = pd.DataFrame(y)
    return df_x, df_y


# process the feature data using the histogram method
def process_data(feature, dimension, size):
    processed_list = []
    for index, row in feature.iterrows():
        processing_list = []
        for n in range(dimension):
            if dimension == 36:
                processing_list.append(row[n + 36] // size)
            else:
                processing_list.append(row[n] // size)
        processed_list.append(processing_list)
    df_feature = pd.DataFrame(processed_list)
    return df_feature


# load the dataset, returns train and test x and y elements
def load_dataset(data_type, prefix, train_files, test_files, dimension, size):
    # load all train
    trainx, trainy = load_dataset_group(data_type, 'train/', prefix, train_files)
    print(trainx.shape, trainy.shape)
    # process the feature data
    trainx = process_data(trainx, dimension, size)
    print(trainx.shape, trainy.shape)
    # load all test
    testx, testy = load_dataset_group(data_type, 'test/', prefix, test_files)
    print(testx.shape, testy.shape)
    # process the feature data
    testx = process_data(testx, dimension, size)
    print(testx.shape, testy.shape)
    # flatten y
    trainy = trainy.values.flatten()
    testy = testy.values.flatten()
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)
    return trainx, trainy, testx, testy


def main():
    # User can change the name these training or testing files
    train_files = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    test_files = ['S7']

    # data_type can be 'Processed_PatientSpace' or 'Processed_RawXY'
    data_type = 'Processed_RawXY'
    # path
    path = '/Users/wangyan/Desktop/Research/Experiment_2/'

    # User can adjust the number of windows for the histogram
    window_num = 40

    if data_type == 'Processed_PatientSpace':
        dimension = 36
        # the max distance is calculated by sqrt(1000^2 + 2000^2)
        max_dist = 4472.23595
    else:
        dimension = 72
        max_dist = 4000

    # calculate the size of each window
    size = max_dist / window_num

    # load dataset
    trainx, trainy, testx, testy = load_dataset(data_type, path, train_files, test_files, dimension, size)

    # create neural network
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

    # fit the model
    mlp.fit(trainx, trainy)

    # make predictions
    yhat = mlp.predict(testx)

    # evaluate predictions and print the results
    print(confusion_matrix(testy, yhat))
    print(classification_report(testy, yhat))
    print(accuracy_score(testy, yhat))


main()
