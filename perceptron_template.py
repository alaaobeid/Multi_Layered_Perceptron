import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, max_iterations=20, learning_rate=0.1):
        self.no_inputs = no_inputs
        self.weights = (np.ones(no_inputs) / no_inputs)
        for i in range(len(self.weights)):
            if i % 2 == 0:
                self.weights[i]=-self.weights[i]
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    #=======================================#
    # Prints the details of the perceptron. #
    #=======================================#
    def print_details(self):
        print("No. inputs:\t" + str(self.no_inputs))
        print("Max iterations:\t" + str(self.max_iterations))
        print("Learning rate:\t" + str(self.learning_rate))

    #=========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    #=========================================#
    def predict(self, inputs):
        activation= np.dot(self.weights,inputs)
        if activation>0:
            return 1
        return 0

                                        ####PART 4####
                                        
    def predict_sigmoid(self, inputs):
        activation= np.dot(self.weights, inputs)
        output=1/(1 + np.exp(-activation))
        if output <= 0.99:
            return 0
        return 1
    #======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    #======================================#
    def train(self, training_data, labels):
        assert len(training_data) == len(labels)
        max_iterations=self.max_iterations
        for i in range(max_iterations):
            for data, label in zip(training_data,labels):
                prediction=self.predict_sigmoid(data)
                self.weights=self.weights + self.learning_rate*(label-prediction)*data
        return
    
                                        ####PART 2####
                                        
    def batch_learning(self, training_data, labels):
        assert len(training_data) == len(labels)
        for i in range(self.max_iterations):
            weights_update = np.zeros(self.no_inputs)
            for data, label in zip(training_data, labels):
                prediction = self.predict_sigmoid(data)
                weights_update += self.learning_rate*(label - prediction)*data
                self.weights += weights_update / len(labels)
        return 
                
    #=========================================#
    # Tests the prediction on each element of #
    # the testing data. Prints the precision, #
    # recall, and accuracy of the perceptron. #
    #=========================================#
    def test(self, testing_data, labels):
        assert len(testing_data) == len(labels)
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        p = 0
        tp = 0
        fn = 0
        for data,label in zip(testing_data,labels):
            prediction=self.predict_sigmoid(data)
            if label == True:
                p += 1
            if label == True and prediction == 1:
                tp += 1
            if label == False and prediction == 1:
                fn += 1
            accuracy=accuracy+1-abs(label-prediction)
            print("Expected=%d, Predicted=%d" % (label, prediction))
        accuracy=accuracy/len(labels)
        precision = tp / (tp + (p-tp))
        recall = tp / (tp+fn)
        print("Accuracy:\t"+str(accuracy))
        print("Precision:\t"+str(precision))
        print("Recall:\t"+str(recall))
                            
            
                                                ####PART 5####
                                                
    def print_weights(self):
        h=self.weights[0:784]
        plt.imshow(h.reshape(28,28),cmap='gray')