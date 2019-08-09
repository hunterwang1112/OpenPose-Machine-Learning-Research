# OpenPose-Machine-Learning

Apply Machine learning on the OpenPose data from Camera 2. The machine learning process uses the classifiers from the Scikit-Learn Package. For basic machine learning, the scripts are based on 8 different classifiers. For deep learning with Neural Network, the scripts are based on the MLP Classifier. Besides, I also tried different techniques, but most of them are proved not helpful. Still, these scripts are kept in this repo.

For all the machine learning scripts, training set is data from subjects 1-6, while testing set is data from subject 7.

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
