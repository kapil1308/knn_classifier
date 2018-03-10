from sklearn.datasets import load_breast_cancer         # Import the Breast Cancer Wisconsin (Diagnostic) Data Set
from sklearn.neighbors import KNeighborsClassifier      # Importing the KNNClassifier algorithm to be used in the project
from sklearn.model_selection import train_test_split    # Importing to divide the dataset into two parts - training and test

import matplotlib.pyplot as plt   # For plotting a graph


cancer = load_breast_cancer()  # Loading the cancer dataset

print(cancer.target_names)     # Print the target names into which the data is classified. In this case two - malignant and benign

print(cancer.DESCR)            # Print the description of the cancer dataset

# Dividing the data
# set into two parts - One is the training set to train the algorithm on and the second the testing set
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,stratify=cancer.target, random_state=66)


training_accuracy = []   # Store the accuracy values of all the training data
test_accuracy = []       # Store the accuracy values of all the test data

# Running a For loop to calcualte accuracy values of number of nearest neighbours ranging from 1 to 11

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train,y_train)
    training_accuracy.append(clf.score(x_train,y_train))
    test_accuracy.append(clf.score(x_test,y_test))

# Plotting the accuracies of both the training as well as the testing data to get the optimal accuracy for training set

plt.plot(neighbors_settings, training_accuracy, label="Accuracy of the training set")
plt.plot(neighbors_settings, test_accuracy, label="Accuracy of the test set")
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbours')
plt.show()




