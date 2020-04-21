# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:49:59 2020

@author: Aladdin Tahir
"""
#iterate over the test data and print the number of the activated perceptron and the prediction
import sys
p[0].print_weights()
for t in testing_data:
    for i in range(10):   
        prediction = p[i].predict(t)
        if prediction == 1:

            dat = t[1:].reshape((28, 28))
            for k in range(28):
                for j in range(28):
                    if dat[k][j]>0:
                        sys.stdout.write("#")
                    else:
                        sys.stdout.write(".")
                    sys.stdout.flush()
                sys.stdout.write("\n")
            
            print("Perceptron Activated: ",i)
            print("Number Detected: ", i)
            time.sleep(0.2)
            
