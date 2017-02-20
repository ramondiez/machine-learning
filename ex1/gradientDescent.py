'''
Created on 10 feb. 2017

@author: fara
'''
from computeCost import computeCost
import numpy as np
from show import show

def gradientDescent(X, y, theta, alpha, num_iters):
    '''GRADIENTDESCENT Performs gradient descent to learn theta
       theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
       taking num_iters gradient steps with learning rate alpha
    '''
    # Initialize some useful values
    m = y.size; # number of training examples
    J_history = [];
    for iter in range(num_iters):
    
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
         
        theta_change = alpha * (X.T.dot(X.dot(theta)-y))/m #Vectorized implementation in python 
        theta = theta - theta_change;
        
        # ============================================================
        # Save the cost J in every iteration    
        J_history.append(computeCost(X, y, theta))  
        
    return theta,J_history 
            
    
    
    