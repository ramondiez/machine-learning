'''
Created on 20 feb. 2017

@author: fara
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from show import show
from predict import predict
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
'''
# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
'''



## Load Data
#  The first two columns contains the X values and the third column contains the label (y).

data = np.loadtxt('ex2data2.txt',delimiter=",")
X = data[:, 0:2]

y = data[:, 2]

#Plot in interactive mode
plt.ion() 
fig, ax = plt.subplots(1, 1)
plotData(X, y);

#Setting the axes labels
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')

plt.legend(['y = 1', 'y = 0'],loc='upper right', fontsize='x-small', numpoints=1) 
plt.show()
plt.pause(0.0001)
raw_input("Program paused. Press Enter to continue.")


''' =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%
'''
# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X[:,0], X[:,1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
Lambda = 1.0

# Compute and display initial cost and gradient for regularized logistic regression
cost = costFunctionReg(initial_theta, X, y, Lambda)

print'Cost at initial theta (zeros): {}'.format(cost)

raw_input("Program paused. Press Enter to continue.")




''' ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%
'''
# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1]))

# Set regularization parameter lambda to 1 (you should vary this)
Lambda = 1.0

# Optimize
result = minimize(costFunctionReg, initial_theta, method='L-BFGS-B',jac=gradientFunctionReg, args=(X, y, Lambda),
                  options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

theta = result.x
cost = result.fun

# Plot Boundary
plotDecisionBoundary(ax,theta, X, y);

#Setting the lambda in the title
fig.suptitle('lambda =  {}'.format(Lambda), fontsize=14, fontweight='bold')

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()
plt.pause(0.0001)
show()

# Compute accuracy on our training set
p = predict(theta, X)
print'Train Accuracy: {}'.format(np.mean(np.where(p == y,1,0)) * 100)

raw_input("Program paused. Press Enter to continue.")

plt.close()




