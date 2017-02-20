'''
Created on 20 feb. 2017

@author: fara
'''

from scipy.special import expit
import numpy as np
from numpy import eye
from show import show
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, Lambda):
    '''
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 
    '''
    # Initialize some useful values
    m = y.size # number of training examples
    
    # You need to return the following variables correctly 
    J = 0
    grad = []    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    
    #Sigmoid function    
    s=sigmoid(X.dot(theta))
    
    #Theta 0 do not need to be normalized    
    normalized=np.ones(theta.shape) - (eye(theta.shape[0])[:,0])     
    theta = theta * normalized;
    
    
    #Calculation of the Cost function
    firstparam = np.sum(-(y * (np.log(s))) - (1 - y) * (np.log(1 - s)))/m
    secondparam = Lambda * np.sum(theta**2)/(2*m)   
    J = firstparam + secondparam;
    
    #Calculation of the gradient with Regularization parameter
    #error = s - y
    #grad = ((X.T.dot(error))/m) + (Lambda*theta)/m;
    
    return J
    
    