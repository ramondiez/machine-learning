'''
Created on 10 feb. 2017

@author: fara
'''

import numpy as np
import matplotlib.pyplot as plt

from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from show import show


'''
%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     warmUpExercise.py
%     plotData.py
%     gradientDescent.py
%     computeCost.py
%     gradientDescentMulti.py
%     computeCostMulti.py
%     featureNormalize.py
%     normalEqn.py
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%
'''

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m 
print 'Running warmUpExercise ...'
print '5x5 Identity Matrix:'
warmUpExercise()
raw_input("Program paused. Press Enter to continue.")


# ======================= Part 2: Plotting =======================
print 'Plotting Data ...'
data = np.loadtxt('ex1data1.txt', delimiter=',')

#The lines below creates X and y as 2D matrix instead of 1D. Otherwise use X = data[:, 0]; y = data[:, 1]
X = data[:, [0]]; y = data[:, [1]]
m = data.shape[0] # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
# Setting the plot in interactive mode
plt.ion() 
fig, ax = plt.subplots(1, 1)
fig.suptitle('Exercise 1', fontsize=14, fontweight='bold')
plotData(ax,X,y) 
plt.show()
plt.pause(0.0001)
raw_input("Program paused. Press Enter to continue.")



# =================== Part 3: Gradient descent ===================
print'Running Gradient Descent ...'


# Add a column of ones to x
X = np.vstack(zip(np.ones(m),X))

# initialize fitting parameters as a 2D
theta = np.zeros((2,1)) 

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
J = computeCost(X, y, theta)
print 'cost: %0.4f ' % J


# run gradient descent
[theta,Jhistory] = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print 'Theta found by gradient descent: '
print '%s %s \n' % (theta[0], theta[1])

# Plot the linear fit
ax.plot(X[:, 1], X.dot(theta), **{'color':'c','label':'Linear regression'})
plt.legend(loc='upper right', fontsize='x-small', numpoints=1)
plt.show()
plt.pause(0.0001)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5],None).dot(theta)
predict2 = np.array([1, 7],None).dot(theta)
print 'For population = 35,000, we predict a profit of {}'.format(predict1*10000)
print 'For population = 70,000, we predict a profit of {}'.format(predict2*10000)

raw_input("Program paused. Press Enter to continue.")
plt.close()


##TODO CONTOUR






