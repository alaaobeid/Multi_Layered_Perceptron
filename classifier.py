
#Perceptron intialization and loading data
from perceptron_template import *
import numpy as np
train=np.genfromtxt('mnist_train.csv', delimiter=',') 
test=np.genfromtxt('mnist_test.csv', delimiter=',')
training_data = [ np.append([1],d[1:]) for d in train]
#Training perceptron and testing it for the digit 7
train_labels_seven = [d[0]==7 for d in train]
testing_data = [ np.append([1],d[1:]) for d in test]
test_labels_seven = [d[0]==7 for d in test]
perc=Perceptron(28*28+1)
perc.train(training_data,train_labels_seven)
perc.test(testing_data,test_labels_seven)


import sys
import time
test_labels = [d[0] for d in test]
#Creating a multi-layered perceptron as a list of perceptrons
p = list()
for i in range(10):
    p.append(Perceptron(28*28+1))
train_label_list=list()
#training each perceptron on a digit from 0 to 9
for i in range(10):
    training_labels= [d[0]==i for d in train]
    train_label_list.append(training_labels)
for i in range(10):
    p[i].batch_learning(training_data,train_label_list[i])
#testing each perceptron in the list on the digit it was trained to predict
test_label_list=list()
for i in range(10):
    testing_labels= [d[0]==i for d in test]
    test_label_list.append(testing_labels)
for i in range(10):
    p[i].test(testing_data,test_label_list[i])
for data,label in zip(testing_data,test_labels):
    for i in range(10):   
        prediction = p[i].predict(data)
        if prediction == 1:
            print("Predicted: ",i," Actual: ",int(label))
            time.sleep(0.2)
                                    
                                            



