from .AutoDiff import AutoDiff as AD 
from .ElementFunction import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
#from mpl_toolkits import mplot3d

def grad_descent(function, x0, alpha=1e-5, max_iter=int(1e6), converge_threshold=1e-8, verbose=False):
    """Gradient descent algorithm for finding the minimum of a function.
    
    Input Arguments:
    =================
    function: a mathematical function f(x): R^n -> R, 
    where the output is scalar (we can only take gradient of scalar function)
    and it must be a int/float/np.number type
    x0: an initial value of the function variable (x), 
    must be a 1D np.array for multiple variables,
    and a scalar for single variable
    alpha: learning rate
    max_iter: maximum number of iterations
    converge_threshold: threshold for convergence
    verbose: whether to print the intermediate results and visualize the process
    
    Return Arguments:
    =================
    x: the input which gives the minimum of the function f(x)
    or the value returned when the maximum number of iterations is reached
    """
    f = AD(function)
    # check if user inputs a scalar function
    if f.dim_f > 1:
        raise TypeError("The function output must be a scalar.")
    # check the initial guess dim
    if f.dim_x == 1:
        if not isinstance(x0, (int, float, np.number)):
            raise TypeError("The initial guess must be a scalar since your function has single variable.")
    else:
        # check if the initial guess is np.array
        if not isinstance(x0, np.ndarray):
            raise TypeError("The initial guess must be a 1D np.array for multi-variable function.")
        # check if the initial guess is 1D array for multi-variable function
        if x0.ndim != 1:
            raise TypeError("The initial guess must be a 1D np.array for multi-variable function.")
        if not all(isinstance(x0_i, (int, float, np.number)) for x0_i in x0):
            raise TypeError("The initial guess must be a 1D np.array of numbers for multi-variable function.")

    # multi-variable function
    if f.dim_x > 1:
        # add a dimension to x0 to 
        # let the dim to (1,n) match with the f.grad() output
        x = np.array([x0])
        for i in range(max_iter):
            x_prev = x
            x = x - alpha*f.grad(x[0]) # use x[0] to get the 1D array
            if i%5000==0 and verbose:
                print(f"Iteration {i}: x={x[0]}, f(x)={f(x[0])}")
            if la.norm(x-x_prev) < converge_threshold:
                # get the 1D array
                return x[0]
        print("Maximum number of iterations reached.")
        return x[0]

    # single-variable function
    if f.dim_x == 1:
        x = x0
        for i in range(max_iter):
            x_prev = x
            x = x - alpha*f.grad(x)
            if i%300==0 and verbose:
                print(f"Iteration {i}: x={x:.7f}, f(x)={f(x):.7f}")
                ## show animation for 1D function
                x_demo = np.linspace(-4,5,100)
                plt.plot(x_demo, [f(x) for x in x_demo], 'b-')
                plt.plot(x, f(x), 'ro')
                plt.pause(0.25)
                plt.cla()
            if la.norm(x-x_prev) < converge_threshold:
                return x
        print("Maximum number of iterations reached.")
        return x

