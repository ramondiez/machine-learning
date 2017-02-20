'''
Created on 15 feb. 2017

@author: fara
'''

from show import show
def computeCostMulti(X, y, theta):
    '''%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    %   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y
    '''
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variables correctly 
    J = 0
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    
    #Cost function with multivariable
    #prediction = sum((X * theta - y) .^2);
    #J = prediction/(2*m);
    
    # This both methods are analogous for MULTI-VARIABLE
    
    # Another vectorized form to compute the cost function
    #J = ((X*theta -y)' * (X*theta -y))/(2*m);
    JMatrix = ((X.dot(theta) - y).T.dot((X.dot(theta) - y)))/(2*m)
    J = JMatrix.item() #Return just the value, not the matrix
    
    # =========================================================================
    
    return J