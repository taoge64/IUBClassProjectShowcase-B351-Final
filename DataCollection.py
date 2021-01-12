import numpy as np
import csv
import time
# it use the load date code in a5, but make many changes to fit our dataset (african_crises)
from KNNClassifier import *
from NeuralNetworkClassifier import *
from PerceptronClassifier import *
from ClusterClassifier import *
from RandomForestClassifier import *
def load_data(filename):
    # Create list to hold raw data
    data_raw = []
    with open(filename, 'r') as f:
        data_raw = list(csv.reader(f, delimiter=','))

    # Try to check if there is any empty list from final line, delete if it exists
    try:
        data_raw.remove([])
    except:
        pass
    # Only consider numerous values for data points
    data_points = np.array([[x[3], x[4], x[5], x[6],x[7],x[8],x[9],x[10],x[11],x[12]] for x in data_raw], dtype='float32')

    # A point is labelled positive if its final column is ">50K"
    data_labels = np.array([1 if x[13] == 'crisis' else 0 for x in data_raw], dtype='int')
    return data_points, data_labels

def KNN_classifier(k):
        first = time.time()
        TRAINING_LENGTH = 859 # we'll train on 859 points
        VALIDATION_LENGTH = 200 # we'll us 200 points as validation data

        points, labels = load_data('african_crises1.csv')
        train_points, train_labels = points[:TRAINING_LENGTH], labels[:TRAINING_LENGTH]
        validation_points, validation_labels = points[TRAINING_LENGTH:TRAINING_LENGTH + VALIDATION_LENGTH], labels[TRAINING_LENGTH:TRAINING_LENGTH + VALIDATION_LENGTH]

        basic_knn = KNN_Classifier(k)

        basic_knn_correct = 0

        for point, label in zip(validation_points, validation_labels):
            if basic_knn.classify(point, train_points, train_labels) == label:
                basic_knn_correct += 1
        print(f'{k}NN classifier got {basic_knn_correct} correct!')

        print(f'1NN classifier accuracy: {basic_knn_correct / float(len(validation_labels)) * 100.0}%')
        print(f'3NN classifier accuracy: {basic_knn_correct / float(len(validation_labels)) * 100.0}%')
        print('The time of ', k, 'NN_classifier is ', time.time() - first)

def Cluster_classifier():
    first = time.time()
    TRAINING_LENGTH = 859  # we'll train on 859 points
    VALIDATION_LENGTH = 200  # we'll us 200 points as validation data

    points, labels = load_data('african_crises1.csv')
    train_points, train_labels = points[:TRAINING_LENGTH], labels[:TRAINING_LENGTH]
    validation_points, validation_labels = points[TRAINING_LENGTH:TRAINING_LENGTH + VALIDATION_LENGTH], labels[
                                                                                                        TRAINING_LENGTH:TRAINING_LENGTH + VALIDATION_LENGTH]

    cluster_classfier=Cluster_Classifier(2)
    cluster_correct = 0

    for point, label in zip(validation_points, validation_labels):
        if cluster_classfier.classify(point,train_points) == label:
            cluster_correct += 1
    print(f'Cluster classifier got {cluster_correct} correct!')
    print(f'Cluster classifier accuracy: {cluster_correct/ float(len(validation_labels)) * 100.0}%')
    print('The time of Cluster classifier is ', time.time() - first)

def Perceptron_classifier():
    first = time.time()
    TRAINING_LENGTH = 859  # we'll train on 859 points
    VALIDATION_LENGTH = 200  # we'll us 200 points as validation data

    points, labels = load_data('african_crises1.csv')
    train_points, train_labels = points[:TRAINING_LENGTH], labels[:TRAINING_LENGTH]
    validation_points, validation_labels = points[TRAINING_LENGTH:TRAINING_LENGTH + VALIDATION_LENGTH], labels[
                                                                                                        TRAINING_LENGTH:TRAINING_LENGTH + VALIDATION_LENGTH]

    # initiallize the percetron classifer
    perceptron = Perceptron_Classifer()
    # initialize the correct number of perceptron
    perceptron_correct = 0
    for point, label in zip(validation_points, validation_labels):
        # classify each point and match with label
        if perceptron.classify(point, train_points, train_labels) == label:
            perceptron_correct += 1
    # result showing
    print(f'Perceptron classifier got {perceptron_correct} correct!')
    print(f'Perceptron classifier accuracy: {perceptron_correct / float(len(validation_labels)) * 100.0}%')
    print('The time of Perceptron is ', time.time() - first)
# get directly from the RandomForestClassifier
def RandomForest_classifier():
    randomforest=RandomForest_Classifier()
    return randomforest.classify()
# get directly from NeuralNetwork_Classifier
def NeuralNetwork_classifier():
    neuralnetwork=NeuralNetworkClassifier()
    return neuralnetwork.classify()

if __name__ == "__main__":
    #calling in each method for the accuracy result
   KNN_classifier(1)
   KNN_classifier(3)
   Cluster_classifier()
   Perceptron_classifier()
   print('The Accuracy of Random Forest is ', RandomForest_classifier()*100)
   print('The Accuracy of Neural Network is ', NeuralNetwork_classifier()*100)
