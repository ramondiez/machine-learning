'''
Created on 24 feb. 2017

@author: fara
'''

import numpy as np
from scipy.optimize import minimize,fmin_cg
from lrCostFunction import lrCostFunction
from gradientFunctionReg import gradientFunctionReg
from show import show


def oneVsAll(X, y, num_labels, Lambda):
    '''
    %ONEVSALL trains multiple logistic regression classifiers and returns all
    %the classifiers in a matrix all_theta, where the i-th row of all_theta 
    %corresponds to the classifier for label i
    %   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    %   logistic regression classifiers and returns each of these classifiers
    %   in a matrix all_theta, where the i-th row of all_theta corresponds 
    %   to the classifier for label i
    '''
    # Some useful variables
    m,n = X.shape
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    
    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))
    
    '''
    % ====================== YOUR CODE HERE ======================
    % Instructions: You should complete the following code to train num_labels
    %               logistic regression classifiers with regularization
    %               parameter lambda. 
    %
    % Hint: theta(:) will return a column vector.
    %
    % Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    %       whether the ground truth is true/false for this class.
    %
    % Note: For this assignment, we recommend using fmincg to optimize the cost
    %       function. It is okay to use a for-loop (for c = 1:num_labels) to
    %       loop over the different classes.
    %
    %       fmincg works similarly to fminunc, but is more efficient when we
    %       are dealing with large number of parameters.
    %
    % Example Code for fmincg:
    %
    %     % Set Initial theta
    %     initial_theta = zeros(n + 1, 1);
    %     
    %     % Set options for fminunc
    %     options = optimset('GradObj', 'on', 'MaxIter', 50);
    % 
    %     % Run fmincg to obtain the optimal theta
    %     % This function will return theta and the cost 
    %     [theta] = ...
    %         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    %                 initial_theta, options);
    %
    '''
      
    y=y.flatten()
        
    for c in range(num_labels):
        initial_theta = np.zeros((n + 1,1))
        
        #show(initial_theta.shape, X.shape, y.shape)
        #c=np.full((m,), c, dtype=int)
        
        #show(c)
        #J=lrCostFunction(initial_theta, X, y==c, Lambda)
        #show(J)
        #result=minimize(fun=lrCostFunction, x0=initial_theta, args=(X, y==c, Lambda), method='TNC', jac=gradientFunctionReg)
        #result = minimize(lrCostFunction, initial_theta, method='CG', args=(X, y==c, Lambda),options={'gtol': 1e-4, 'disp': True, 'maxiter': 500})
        #result = minimize(lrCostFunction, initial_theta, method='CG',jac=gradientFunctionReg, args=(X, y==c, Lambda),options={'gtol': 1e-4, 'disp': False, 'maxiter': 50})
        result = minimize(lrCostFunction, initial_theta, method='L-BFGS-B',jac=gradientFunctionReg, args=(X, y==c, Lambda),options={'maxiter': 50})
        #show(result)
        all_theta[c,:]=result.x
        show("CostFunction: ",result.fun)
        #all_theta[c,:] = fmin_cg(lrCostFunction, initial_theta, fprime=gradientFunctionReg, gtol=1e-05,args=(X, y==c, Lambda),maxiter=500)
    return all_theta
    
    # =========================================================================
    
    
    
