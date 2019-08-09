import pandas as pd
import numpy as np
import statistics
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
    print(np.unique(y))
    df_y = pd.DataFrame(y)
    return df_x, df_y


# process the data to get the median of all the frames in the window for each dimension
def windowing(feature, label, numFrame, dimension):
    processed_list = []
    processing_list = []
    for i in range(dimension):
        feature_list = []
        processing_list.append(feature_list)

    for index, row in feature.iterrows():
        for n in range(dimension):
            if dimension == 36:
                (processing_list[n]).append(row[n + 36])
            else:
                (processing_list[n]).append(row[n])
        if len(processing_list[0]) == numFrame:
            append_list = []
            for j in range(dimension):
                median = statistics.median(processing_list[j])
                append_list.append(median)
                processing_list[j].pop(0)
            processed_list.append(append_list)

    # generate a new dataframe
    df_feature = pd.DataFrame(processed_list)

    # delete the extra label data in the head and tail
    if numFrame // 2 == numFrame / 2:
        label.drop(label.head(numFrame // 2 - 1).index, inplace=True)
        label.drop(label.tail(numFrame // 2).index, inplace=True)
    else:
        label.drop(label.head(numFrame // 2).index, inplace=True)
        label.drop(label.tail(numFrame // 2).index, inplace=True)
    return df_feature, label


# load the dataset, returns train and test x and y elements
def load_dataset(data_type, prefix, train_files, test_files, numFrame, dimension):
    # load all train
    trainx, trainy = load_dataset_group(data_type, 'train/', prefix, train_files)
    print(trainx.shape, trainy.shape)
    # process the feature data and label data
    trainx, trainy= windowing(trainx, trainy, numFrame, dimension)
    print(trainx.shape, trainy.shape)
    # load all test
    testx, testy = load_dataset_group(data_type, 'test/', prefix, test_files)
    print(testx.shape, testy.shape)
    # process the feature data and label data
    testx, testy = windowing(testx, testy, numFrame, dimension)
    print(testx.shape, testy.shape)
    # flatten y
    trainy = trainy.values.flatten()
    testy = testy.values.flatten()
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)
    return trainx, trainy, testx, testy


def main():
    # User can change the name of the training or testing files
    train_files = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    test_files = ['S7']

    # data_type can be 'Processed_PatientSpace' or 'Processed_RawXY'
    data_type = 'Processed_RawXY'

    # path of the label, PatientSpace and RawXY
    path = '/Users/wangyan/Desktop/Research/Experiment_2/'

    # Use can number of frames in a window
    numFrame = 30

    # set the dimensions for two kinds of feature data
    if data_type == 'Processed_PatientSpace':
        dimension = 36
    else:
        dimension = 72

    # load dataset
    trainx, trainy, testx, testy = load_dataset(data_type, path, train_files, test_files, numFrame, dimension)

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
