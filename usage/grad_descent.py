import sys
sys.path.append('../')
from AutoDiff.AutoDiff import AutoDiff as AD
from AutoDiff.feature import grad_descent
from AutoDiff.ElementFunction import *
import numpy as np
# import pdb

""" gradient descent """

## demo of 1D function
def demo_1D():
    f = lambda x: (x-1)**2
    x0 = 4
    x = grad_descent(f, x0, alpha=1e-3, max_iter=int(1e6), verbose=True)
    print(f"x={x:.3f} gives the minimum value of f(x)={f(x):.3f}")

def demo_1D_quartic():
    f = lambda x: -10*x**2 + x**4 - x 
    x0 = 4
    x = grad_descent(f, x0, alpha=1e-5, verbose=True)
    print(f"x={x:.3f} gives the minimum value of f(x)={f(x):.3f}")

def demo_2D():
    f = lambda x, y: (sin(x-0.1))**2+(cos(y))**2
    # create an AutoDiff object to get function value 
    ADf = AD(f)
    x0 = np.array([0.5, 0.2])
    x = grad_descent(f, x0, alpha=1e-4, converge_threshold=1e-8, verbose=True)
    print(f"x={x} gives the minimum value of f(x)={ADf(x):.3f}")


"""run the gradient descent demo """
demo_1D()
# demo_1D_quartic()
# demo_2D()


print(""" ------------reduce the loss demo--------------  """)

""" reduce the loss demo 
The following shows how to use the gradient descent to reduce the loss of a logistic regression model."""

def sigmoid(x):
    return 0.5 * (tanh(x / 2.) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(w0, w1, w2):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(np.array([w0,w1,w2]), inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(log(label_probabilities))

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

# Define a function that returns gradients of training loss using AutoDiff.
training_gradient_fun = AD(training_loss).grad

# Optimize weights using gradient descent.
# You will see the loss is reduced significantly.
weights = np.array([0.0, 0.0, 0.0])
print("Initial loss:", training_loss(*weights))
for i in range(1000):
    weights -= training_gradient_fun(weights)[0] * 0.01

print("Trained loss:", training_loss(*weights))

