'''
Created on 24 feb. 2017

@author: fara
'''

from sigmoid import sigmoid
from numpy import eye
from show import show
import numpy as np

def lrCostFunction(theta, X, y, Lambda):
    '''
    %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    %regularization
    %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 
    '''
    # Initialize some useful values
    m = len(y) # number of training examples
    #show(y)
    # You need to return the following variables correctly 
    J = 0
    
    '''
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    %
    % Hint: The computation of the cost function and gradients can be
    %       efficiently vectorized. For example, consider the computation
    %
    %           sigmoid(X * theta)
    %
    %       Each row of the resulting matrix will contain the value of the
    %       prediction for that example. You can make use of this to vectorize
    %       the cost function and gradient computations. 
    %
    % Hint: When computing the gradient of the regularized cost function, 
    %       there're many possible vectorized solutions, but one solution
    %       looks like:
    %           grad = (unregularized gradient for logistic regression)
    %           temp = theta; 
    %           temp(1) = 0;   % because we don't add anything for j = 0  
    %           grad = grad + YOUR_CODE_HERE (using the temp variable)
    %
    '''
            
    s=sigmoid(X.dot(theta))
    
    #Theta 0 do not need to be normalized    
    normalized=np.ones(theta.shape) - (eye(theta.shape[0])[:,0])     
    theta = theta * normalized
    
    
    #Calculation of the Cost function
    firstparam = np.sum(-(y * (np.log(s))) - (1 - y) * (np.log(1 - s)))/m
    secondparam = Lambda * np.sum(theta**2)/(2*m)   
    J = firstparam + secondparam;
    #show("CostFunction: ",J)
    
    return J
    
    
    
    
   
