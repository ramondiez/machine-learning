'''
Created on 17 feb. 2017

@author: fara
'''

import numpy as np
def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization
    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m = len(y)   # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    
    s=1/(1+np.exp(X.dot(theta)))
    
    #Flatten s to subtract afterwards to y
    s=s.flatten() 
    #Vectorized version of the gradian descent
    error= s - y
    grad = (X.T.dot(error))/m
    # =============================================================
    

    return grad