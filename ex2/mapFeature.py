'''
Created on 20 feb. 2017

@author: fara
'''
import numpy as np
from show import show
def mapFeature(X1, X2):
    '''
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of 
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #
    '''
    #out = []
    degree = 6;
    '''out = ones(size(X1(:,1)));
    for i in range(degree):
        for j in range(i):
            out(:, end+1) = (X1.^(i-j)).*(X2.^j);
        end
    end
    '''
    #Create an array of ones
    out=np.ones((np.size(X1)))    
    
    for i in range(1,degree+1):
        for j in range(0,i+1):
            #Stack a column per iteration                     
            out=np.vstack((out,X1**(i-j)*(X2**j)))
    #Transpose the array 118x28
    return out.T