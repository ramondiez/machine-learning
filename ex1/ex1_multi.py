'''
Created on 15 feb. 2017

@author: fara
'''
import numpy as np
import matplotlib.pyplot as plt
from show import show

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
'''
%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization
'''
# ================ Part 1: Feature Normalization ================

#%% Clear and Close Figures
#clear ; close all; clc

print'Loading data ...'

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2];
show("X.shape",X.shape)
y = data[:, [2]];
show("Y.shape",y.shape)
m = data.shape[0];
#show(X[1:10,0])
# Print out some data points
print'First 10 examples from the dataset: '
print np.column_stack( (X[:10], y[:10]) )
raw_input("Program paused. Press Enter to continue.")

# Scale features and set them to zero mean
print'Normalizing Features ...'

X, mu ,sigma = featureNormalize(X)
'''
show("X",X.shape)
show("mu",mu.shape)
show("sigma",sigma.shape)
'''
raw_input("Program paused. Press Enter to continue.")

# Add intercept term to X
#X = [ones(m, 1) X];
#X = zip(np.ones((m,1)),X)
X = np.concatenate((np.ones((m, 1)), X), axis=1)



# ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print'Running gradient descent ...'

# Choose some alpha value
alpha = 0.01;
num_iters = 400;

# Init Theta and Run Gradient Descent
theta = np.zeros((3,1));

#Calculate the GradientDescent for Multiple Variables
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters);



# Plot the convergence graph
plt.ion()
fig, ax = plt.subplots(1, 1)
ax.plot(J_history ,linewidth=2, **{'color':'r'})
ax.set_ylabel('Cost J')
ax.set_xlabel('Number of iterations')
plt.show()
plt.pause(0.001)
raw_input("Program paused. Press Enter to continue...")

# Display gradient descent's result
print'Theta computed from gradient descent:'
print(' {} '.format(theta));
plt.close()


#Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.


#Add 0 value column to mu matrix
mu=np.concatenate((np.zeros((1, 1)), mu), axis=1)

#Add 1 value column to sigma matrix
sigma=np.concatenate((np.ones((1, 1)), sigma), axis=1)

#Reshape house object to run vectorized way
house=np.array([1,1650,3]).reshape(1,3)

price = ((house-mu)/sigma).dot(theta)

# ============================================================

print'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${}'.format(price.item());


raw_input("Program paused. Press Enter to continue...")


## ================ Part 3: Normal Equations ================

print'Solving with normal equations...'

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#


#Initialize again the values
X = data[:, 0:2]
y = data[:, [2]]
m = y.size;

# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y);

# Display normal equation's result
print'Theta computed from the normal equations: '
print' {} '.format(theta);



# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = np.array([1,1650,3]).reshape(1,3).dot(theta); 


# ============================================================

print'Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n ${}'.format(price);

