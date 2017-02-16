'''
Created on 15 feb. 2017

@author: fara
'''
from computeCostMulti import computeCostMulti
from show import show

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    '''%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    %   taking num_iters gradient steps with learning rate alpha
    '''
    # Initialize some useful values
    m = y.shape[0] # number of training examples

    J_history = [];
    
    for iter in range(num_iters):
    
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.
        #
        
        #h = X * theta;
        #error = h - y;
        #theta_change = alpha * ((X' * error)/m);
        #theta = theta - theta_change;
        
        h = X.dot(theta)
        error = h - y;
        theta_change = alpha * (X.T.dot(error)/m)
        theta = theta - theta_change;
    
    
        # ============================================================
    
        # Save the cost J in every iteration    
        J_history.append(computeCostMulti(X, y, theta));
    
    return theta,J_history