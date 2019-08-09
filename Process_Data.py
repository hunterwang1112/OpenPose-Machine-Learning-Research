import pandas as pd


# load a single file as a DataFrame
def getFile(filepath):
    dataframe = pd.read_csv(filepath, header=0)
    return dataframe


# process the label data and the feature data
def processData(label_df, xy_df, patient_space_df):
    # create lists for later use
    labelName = []
    frameNum = []
    xy_rows = []
    ps_rows = []
    label = []

    # iterate over the raw label data
    for index, row in label_df.iterrows():
        rowNum = []
        # append the label name to "labelName"
        labelName.append(row[0])
        i = row[9]
        while i < row[10] + 1:
            rowNum.append(i)
            i += 1
        # append the interval to 'frameNum'
        frameNum.append(rowNum)

    # process the RawXY data
    j = 0
    k = 0
    for index_xy, row_xy in xy_df.iterrows():
        run = True
        while j < len(frameNum) and run:
            length = len(frameNum[j])
            while k < length and run:
                if row_xy[0] < frameNum[j][k]:
                    run = False
                elif row_xy[0] == frameNum[j][k]:
                    label.append(labelName[j])
                    xy_rows.append(row_xy)
                    run = False
                    k += 1
                elif row_xy[0] > frameNum[j][k]:
                    k += 1
                    run = True
            if run:
                k = 0
                j += 1

    # process the patientSpace Data
    j = 0
    k = 0
    for index_ps, row_ps in patient_space_df.iterrows():
        run = True
        while j < len(frameNum) and run:
            length = len(frameNum[j])
            while k < length and run:
                if row_ps[0] < frameNum[j][k]:
                    run = False
                elif row_ps[0] == frameNum[j][k]:
                    ps_rows.append(row_ps)
                    run = False
                    k += 1
                elif row_ps[0] > frameNum[j][k]:
                    k += 1
                    run = True
            if run:
                k = 0
                j += 1

    xy_dataframe = pd.DataFrame(xy_rows)
    ps_dataframe = pd.DataFrame(ps_rows)
    label_dataframe = pd.DataFrame({'label':label})

    # get rid of first two columns
    xy_dataframe = xy_dataframe.drop(['Frame', 'Timestamp'], axis=1)
    ps_dataframe = ps_dataframe.drop(['Frame', 'Timestamp'], axis=1)
    print(xy_dataframe)
    print(ps_dataframe)
    print(label_dataframe)
    return label_dataframe, xy_dataframe, ps_dataframe


def main():
    # filename can be S1, S2, S3, S4, S5, S6, S7
    filename = 'S6'

    # paths can be different
    label_path = '/Users/wangyan/Desktop/Research/Experiment_2/Raw_Label_Data/'
    rawxy_path = '/Users/wangyan/Desktop/Research/Experiment_2/RawXY/'
    patient_space_path = '/Users/wangyan/Desktop/Research/Experiment_2/PatientSpace/'

    label_df = getFile(label_path + filename + '.csv')
    xy_df = getFile(rawxy_path + filename + '_C2_RawXY.csv')
    patient_space_df = getFile(patient_space_path + filename + '_C2_PatientSpace.csv')
    label, rawxy, patient_space = processData(label_df, xy_df, patient_space_df)

    label.to_csv('/Users/wangyan/Desktop/Research/Experiment_2/Processed_Label/' + filename + '_label.csv',
                 header=False, index=False)
    rawxy.to_csv('/Users/wangyan/Desktop/Research/Experiment_2/Processed_RawXY/' + filename + '_RawXY.csv',
                 header=False, index=False)
    patient_space.to_csv('/Users/wangyan/Desktop/Research/Experiment_2/Processed_PatientSpace/' + filename +
                         '_PatientSpace.csv', header=False, index=False)


main()
