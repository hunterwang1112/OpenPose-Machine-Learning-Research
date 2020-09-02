# OpenPose-Machine-Learning

OpenPose Machine Learning Research is part of the project to develop a video pipeline that automatically recognizes clinical procedures during emergency care and sends real-time information of patients to receiving hospitals or emregency medical services. This repository includes my work in the 2019 VUSE summer research program at [Vanderbilt University's School of Engineering](https://engineering.vanderbilt.edu). The primary investigator for the project is [Dr. Bobby Bodenheimer](https://engineering.vanderbilt.edu/bio/robert-bodenheimer), Professor at the [Department of Electrical Engineering and Computer Science](https://engineering.vanderbilt.edu/eecs/).

The research applys various machine learning algorithms on OpenPose data collected on Camera 2 from previous experiments to find which classifier or algorithm works the best. The research mainly uses Scikit-learn, a python-based machine learning package. The first part of the research uses 8 machine learning classifiers. The second part of the research, with deep leanring, uses the MLP classifier. Besides, various techniques are used to improve the efficiency of the algorithms, but most of them are proved not helpful. Still, these scripts are kept in this repository.

Subjects 1-6 are training data; subject 7 is testing data.

## Process Data

### Process_Data.py
Process the RawXY, PatientSpace and label data so that they can be directly used as feature data and label data. Users can change the subject names and paths. 
This outputs Processed_RawXY and Processed_PatientSpace files with 72 dimensions and Processed_Label files with a column vector. The output files are frame by frame.
* Although the Processed_PatientSpace files have 72 dimensions, the first 36 dimensions are binary numbers, so in all the machine learning scripts, I only use the last 36 dimensions for PatientSpace.

## Scikit-Learn Classifiers

Run 8 Scikit-Learn classifirs on Processed_PatientSpace or Processed_RawXY.

### Classifiers_Raw.py

Directly run the 8 classifiers on Processed_PatientSpace or Processed_RawXY.

### Classifiers_Median.py

Incorporate the temporal feature between frames. Take the median of all the frames in a window for each dimension. The medians form new feature data.

### Classifiers_Histogram.py

For PatientSpace, the script divides the distances evenly into some number of categories.
For RawXY, the script divides the X and Y axis evenly into categories.

## Deep Learning with Neural Network

Use the Scikit-Learn MLP Classifier to implement the neural network.

### Neural_Network_Raw.py

Directly run the Neural Network on Processed_PatientSpace or Processed_RawXY.

### Neural_Network_Median.py

Incorporate the temporal feature between frames. Take the median of all the frames in a window for each dimension. The medians form new feature data.

### Neural_Network_Histogram.py

For PatientSpace, the script divides the distances evenly into some number of categories.
For RawXY, the script divides the X and Y axis evenly into categories.

## Other Scripts

### Classifiers_Combine_Data.py

This script applies machine learning on the combined data from Processed_PatientSpace and Processed_RawXY. 
Results show that combining two kinds of data cannot increase the testing accuracy.

### Classifiers_Label_Encoding.py

This script does label encoding on the label data, so the labels are numerical values. 
Theoretically, numerical values should work better, but in this case, encoding the labels actually lowers the testing accuracy.

### Classifiers_Random_Split.py

This script first reads data from all the subjects into a single dataframe and then randomly splits the data into training set and testing set.
This script can generate extremely high testing accuracy, but this is inaccurate.

### Classifiers_Temporal_Feature.py

This script expand the feature data to larger data set which combines the frames before and after a certain frame.
Since the dimension becomes very large, this script runs slowly, but the results show that incorporating temporal feature in this way does increase the testing accuracy.

## Confusion Matrics

[Confusion Matrics](Confusion%20Matrices) for PatientSpace and RawXY are generated and can be accessed in this repo.
