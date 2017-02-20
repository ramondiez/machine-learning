'''
Created on 16 feb. 2017

@author: fara
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit

from show import show
from plotData import plotData
from costFunction import costFunction
from gradientFunction import gradientFunction
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

'''
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
'''
# Initialization


# Load Data
# The first two columns contains the exam scores and the third column contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2] #1 Dimension


# ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.'
#Plot in interactive mode
plt.ion() 
fig, ax = plt.subplots(1, 1)

#Plotting the image
plotData(X,y)

#Setting the axes labels
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')

#Define the legend
plt.legend(['Admitted', 'Not admitted'],loc='upper right', fontsize='x-small', numpoints=1) 
plt.show()
plt.pause(0.0001)
raw_input("Program paused. Press Enter to continue.")




## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You need to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m , n = X.shape

# Add intercept term to x and X_test
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Initialize fitting parameters
initial_theta = np.zeros(n+1)


# Compute and display initial cost and gradient
#show(X.shape, y.shape, initial_theta.shape)
cost = costFunction(initial_theta, X, y)
grad = gradientFunction(initial_theta, X, y)


print'Cost at initial theta (zeros): {}'.format(cost)
print'Gradient at initial theta (zeros): {}'.format(grad)


raw_input("Program paused. Press Enter to continue.")

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost 
res = minimize(costFunction, initial_theta, method='TNC',
               jac=False, args=(X, y), options={'gtol': 1e-3, 'disp': True, 'maxiter': 400})

# Print theta to screen
cost=res.fun
theta=res.x
print'Cost at theta found by fminunc: {}'.format(cost)
print'theta: {}'.format(theta);

# Plot Boundary
plotDecisionBoundary(ax,theta, X, y)

# Labels and Legend
#plt.legend(['Admitted', 'Not admitted'],loc='upper right', fontsize='x-small', numpoints=1) 
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
plt.pause(0.0001)
show()

raw_input("Program paused. Press Enter to continue.")
plt.close()


''' ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 
'''

prob = expit(np.array([1, 45, 85]).dot(theta));
print'For a student with scores 45 and 85, we predict an admission probability of {}'.format(prob)

# Compute accuracy on our training set
p = predict(theta, X);
print'Train Accuracy: {}'.format(np.mean(np.where(p==y)[0].size))

raw_input("Program paused. Press Enter to continue.")
