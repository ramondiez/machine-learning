'''
Created on 20 feb. 2017

@author: fara
'''
import numpy as np
from matplotlib import pyplot as plt
from mapFeature import mapFeature
from show import show


def plotDecisionBoundary(ax,theta, X, y):
    '''
    %PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    %the decision boundary defined by theta
    %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    %   positive examples and o for the negative examples. X is assumed to be 
    %   a either 
    %   1) Mx3 matrix, where the first column is an all-ones column for the 
    %      intercept.
    %   2) MxN, N>3 matrix, where the first column is all-ones
    '''
    
    
    
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 2]),  max(X[:, 2])])
    
        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
    
        # Plot, and adjust axes for better viewing
        ax.plot(plot_x, plot_y)
        
        # Legend, specific for the exercise        
        plt.legend(['Admitted', 'Not admitted'],loc='upper right', fontsize='x-small', numpoints=1) 
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)        
        z = [
                np.array([mapFeature(u[i], v[j]).dot(theta) for i in range(len(u)) for j in range(len(v))])                
            ]
        
        #Reshape to get a 2D array
        z=np.reshape(z, (50, 50))
        
        #Draw the plot      
        plt.contour(u,v,z, levels=[0.0])
        
        

        
        
        
    
    