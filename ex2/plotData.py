'''
Created on 16 feb. 2017

@author: fara
'''
import numpy as np
from matplotlib import pyplot as plt
from show import show

def plotData(X, y):
    '''
    %PLOTDATA Plots the data points X and y into a new figure 
    %   PLOTDATA(x,y) plots the data points with + for the positive examples
    %   and o for the negative examples. X is assumed to be a Mx2 matrix.
    
    % Create New Figure
    figure; hold on;
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Plot the positive and negative examples on a
    %               2D plot, using the option 'k+' for the positive
    %               examples and 'ko' for the negative examples.
    %
    '''
    # Find Indices of Positive and Negative Examples
    
    #Retrieve an array of index where y==1
    pos=np.nonzero(y==1)[0]    
    
    #Retrieve an array of index where y==0
    neg=np.nonzero(y==0)[0]   
    
    # Plot Examples    
    #Plotting positive examples
    plt.plot(X[:,0][[pos]],X[:,1][[pos]],'+', markersize=7, markeredgecolor='black', markeredgewidth=2)
    #Plotting negative examples
    plt.plot(X[:,0][[neg]],X[:,1][[neg]],'o', markersize=7, markeredgecolor='black', markerfacecolor='yellow')
    
    
    
    
   