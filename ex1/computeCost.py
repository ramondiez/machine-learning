'''
Created on 10 feb. 2017

@author: fara
'''

import numpy as np
from show import show

def computeCost(X, y, theta):
    '''COMPUTECOST Compute cost for linear regression
       J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    '''
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variables correctly 
    J = 0;
    #show(np.shape(X),np.shape(theta),np.shape(y))
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #                You should set J to the cost.
    
    
    #Cost function calculation with regular formula.       
    J = np.sum((X.dot(theta) - y)**2)/(2*m)    #Python
    
    
    #Compute host function with a Vectorized version       
    #J = (X.dot(theta) - y).T.dot(X.dot(theta) -y ) / (2*m)  #Ptyhon
    return J 
    
    # =========================================================================
    
    