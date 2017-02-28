'''
Created on 28 feb. 2017

@author: fara
'''

import numpy as np

from sigmoid import sigmoid
from show import show


def predict(Theta1, Theta2, X):
    '''
    %PREDICT Predict the label of an input given a trained neural network
    %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    %   trained weights of a neural network (Theta1, Theta2)
    '''
    # Useful values
    m, n = X.shape
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the max
    #       function can also return the index of the max element, for more
    #       information see 'help max'. If your examples are in rows, then, you
    #       can use max(A, [], 2) to obtain the max for each row.
    #
    
    
    #Concatenate ones to X 
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    
    #Apply sigmoid function
    activation_hidden=sigmoid(X.dot(Theta1.T));
    
    # Add intercept term to activation_hidden 
    activation_hidden = np.concatenate((np.ones((m, 1)), activation_hidden), axis=1)
    
    #Apply sigmoid function to activation layer
    hypothesis=sigmoid(activation_hidden.dot(Theta2.T))
    
    #As the Thetas are calulated based on Octave, we add 1 to index
    p=np.argmax(hypothesis,axis=1)+1
    
    # Map 10 values to 0
    p[p==10]=0
    
    return p
        
    

