'''
Created on 15 feb. 2017

@author: fara
'''

import numpy as np
from show import show

def featureNormalize(X):
    '''%FEATURENORMALIZE Normalizes the features in X 
    %   FEATURENORMALIZE(X) returns a normalized version of X where
    %   the mean value of each feature is 0 and the standard deviation
    %   is 1. This is often a good preprocessing step to do when
    %   working with learning algorithms.
    
    % You need to set these values correctly
    X_norm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: First, for each feature dimension, compute the mean
    %               of the feature and subtract it from the dataset,
    %               storing the mean value in mu. Next, compute the 
    %               standard deviation of each feature and divide
    %               each feature by it's standard deviation, storing
    %               the standard deviation in sigma. 
    %
    %               Note that X is a matrix where each column is a 
    %               feature and each row is an example. You need 
    %               to perform the normalization separately for 
    %               each feature. 
    %
    % Hint: You might find the 'mean' and 'std' functions useful.
    %       
    '''
    X_norm, mu, sigma=0,0,0
    
    # returns a vector with the avg of each feature maintaining the dimensions   
    mu = np.mean(X,axis=0,keepdims=True)    
      
    # returns a vector with the std of each feature maintaining the dimensions   
    # ddof= 1 normalize with N-1, provides the square root of the best unbiased estimator of the variance 
    sigma = np.std(X,axis=0,ddof=1,keepdims=True)
    
    # returns a column size    
    m = X.shape[0]    
    
    # Creates a matrix with column vector * mu    
    mu_matrix = np.ones((m,1)).dot(mu) 
    
    # Creates a matrix with column vector * sigma    
    sigma_matrix = np.ones((m,1)).dot(sigma)
    
    #X_norm = (X - mu_matrix) ./ sigma_matrix;
    X_norm = (X - mu_matrix) / sigma_matrix
       
    return X_norm, mu, sigma  
    