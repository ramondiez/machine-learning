'''
Created on 20 feb. 2017

@author: fara
'''

from scipy.special import expit
import numpy as np
from show import show
def predict(theta, X):
    '''
    %PREDICT Predict whether the label is 0 or 1 using learned logistic 
    %regression parameters theta
    %   p = PREDICT(theta, X) computes the predictions for X using a 
    %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    '''
    m = X.shape[0] # Number of training examples
    
    # You need to return the following variables correctly
    p = []
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters. 
    #               You should set p to a vector of 0's and 1's
    #
    
    [p.append(1) if expit(X.dot(theta))[i] >= 0.5 else p.append(0) for i in range(m)]
    return p
    