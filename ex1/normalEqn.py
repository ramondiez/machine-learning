'''
Created on 16 feb. 2017

@author: fara
'''

import numpy as np

def normalEqn(X, y):
    '''
    %NORMALEQN Computes the closed-form solution to linear regression 
    %   NORMALEQN(X,y) computes the closed-form solution to linear 
    %   regression using the normal equations.
    '''
    #theta = zeros(size(X, 2), 1);
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #
    
   
    
    #theta = pinv(X' * X) * X' * y;
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
    
    
    
    # ============================================================
    
    