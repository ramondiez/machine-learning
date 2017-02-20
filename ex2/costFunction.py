'''
Created on 17 feb. 2017

@author: fara
'''
from scipy.special import expit
import numpy as np
from show import show

def costFunction(theta, X, y):
    '''
    %COSTFUNCTION Compute cost and gradient for logistic regression
    %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    %   parameter for logistic regression and the gradient of the cost
    %   w.r.t. to the parameters.
    '''
    # Initialize some useful values
    m = y.size; # number of training examples    
    # You need to return the following variables correctly 
    J = 0;
    #grad = zeros(size(theta));
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #
    
    #expit == sigmoid function
    s=expit(X.dot(theta)) 
    #s=1/(1+np.exp(-X.dot(theta)))  
    
    
    #Cost function calculation
    prediction = np.sum(-(y * (np.log(s))) - (1-y) * (np.log(1-s)))
    J = prediction/m  
    #J=(-y.T.dot((np.log(s))) - (1-y).T.dot(np.log(1-s)))/m
    
    return J