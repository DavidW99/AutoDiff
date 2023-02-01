"""
This file contains class AutoDiff and RAutoDiff.
AutoDiff is for forward mode, and it contains two methods:
___call___() is for getting the value of the function.
grad() is for getting the derivative of the variables via forward AD mode.
"""
from .DualNumber import DualNumber
from .ElementFunction import *
import numpy as np
from inspect import signature
from copy import deepcopy
import types
import warnings

def exam_user_function(function):
    """Check if the user defined function is a valid mathematical function for AutoDiff.
        
        Input Arguments:
        =================
        function: a mathematical function (lambda function or def function)
        it should follow the requirements below:
        f: x -> y, where dimension of x is dim_x > 0, dimension of y is dim_y > 0
        the number of arguments in your defined function should equal to dim_x
        the output of your defined function should be a number/a list of numbers/a numpy array of numbers, multi-dimensional output is not supported.
        example:
        f1 = lambda x, y: cos(sin(x*y)) + y     # yes
        f2 = lambda x, y: [sin(x*y),x+y,x]      # yes
        f3 = lambda x, y: [sin(x*y),x+y,[1,2]]  # no
        f4 = lambda x: np.array([x**2,x])       # yes
        
        Return Arguments:
        =================
        boolean: True if the function is valid, raise TypeError otherwise
        
        Examples
        =================
        >>> f1 = lambda x, y: sin(x*y)         # 2 to 1 mapping
        >>> exam_user_function(f1)
        True
        >>> f2 = lambda x, y: [sin(x*y),x+y,x] # 2 to 2 mapping
        >>> exam_user_function(f2)
        True
        >>> def f3():
        ...     print(1)
        ... 
        >>> exam_user_function(f3)
        TypeError: The input of your given function is empty
        """
    ## check function type
    if function.__class__ != types.FunctionType:
        raise TypeError("The input is NOT of function type")
    ## get the input dimension
    dim_x = len(signature(function).parameters)
    ## check if the function is a valid mathematical function
    if dim_x == 0:
        raise TypeError("The input of your given function is empty")
    # generate a fake input to get the output dimension
    fake_input = (1,)*dim_x
    output = function(*fake_input)
    if isinstance(output, (list, np.ndarray)):
        # check if it is a 0-d array
        try:
            f_dim = len(output)
        except TypeError:
            raise TypeError("The output of your given function cannot be a 0-d array")
        if all(isinstance(out_i, (int, float, np.number)) for out_i in output):
            if len(output) == 0:
                raise TypeError("The output of your given function is empty")
            return True
        else:
            raise TypeError("The output of your given function contains non-numeric elements/we don't take multi-dimensional output")
    elif isinstance(output, (int, float, np.number)):
        return True
    else:
        raise TypeError("The output of your given function is NOT a valid mathematical function")
        


class AutoDiff():

    def __init__(self, function):
        """Constructor for the AutoDiff class.
        
        Input Arguments:
        =================
        function: a mathematical function
        *** see exam_user_function() for the requirements of the function
        
        Return Arguments:
        =================
        self: Object of the AutoDiff class
        
        Examples
        =================
        f1 = lambda x, y: sin(x*y)         # 2 to 1 mapping
        ad1 = AutoDiff(f1)
        f2 = lambda x, y: [sin(x*y),x+y,x] # 2 to 2 mapping
        ad2 = AutoDiff(f2)
        """
        ## check if the user defined function is a valid mathematical function
        if exam_user_function(function):
            self.function = function
            self.dim_x = len(signature(function).parameters)
            ## get the output dimension
            fake_input = (1,)*self.dim_x
            output = function(*fake_input)
            if isinstance(output, (list, np.ndarray)):
                self.dim_f = len(output)
            else:
                self.dim_f = 1


    def __call__(self, x_vec):
        """Evaluate the function at x_vec
        
        Input Arguments:
        =================
        x_vec: a scalar (float/int/np.number) or a list of scalars

        Return Arguments:
        =================
        output of the function at x_vec
        
        Examples
        =================
        >>> f = AutoDiff(lambda x: sin(x))
        >>> f(1)
        0.8414709848078965
        >>> f([1])
        0.8414709848078965
        >>> f = AutoDiff(lambda x: [sin(x), cos(x)])
        >>> f(1)
        [0.84147098, 0.54030231]
        >>> f([1])
        [0.84147098, 0.54030231]
        >>> f(np.array([1]))
        [0.84147098, 0.54030231]
        >>> f = AutoDiff(lambda x, y: sin(x*y))
        >>> f([1,2])
        0.9092974268256817
        >>> f = AutoDiff(lambda x: np.array([sin(x), cos(x)]))
        >>> f(1)
        array([0.84147098, 0.54030231])
        """

        if self.dim_x == 1:
            # in case user input a length-1 list/np.array
            if isinstance(x_vec, (list, np.ndarray)):
                if len(x_vec) != 1:
                    raise ValueError("Your input dimension does not match the function input dimension")
                if isinstance(x_vec[0], (int, float, np.number)):
                    x_vec = float(x_vec[0])
                else:
                    raise TypeError("Input contains a non-int or non-float") 
                return self.function(x_vec)
            elif isinstance(x_vec, (int, float, np.number)):
                return self.function(float(x_vec))
            else:
                raise TypeError("Input contains a non-int or non-float")
        else:
            try :
                len(x_vec)
            except TypeError:
                raise ValueError("Your input dimension does not match the function input dimension")
            if len(x_vec) != self.dim_x:
                raise ValueError("Your input dimension does not match the function input dimension")
            if isinstance(x_vec, (list, np.ndarray)):
                if all(isinstance(x_i, (int, float, np.number)) for x_i in x_vec):
                    # convert the input to float
                    x_vec = [float(x_i) for x_i in list(x_vec)]
                    # return a n-dim array with rows as the output dimension
                    return self.function(*x_vec)
                else:
                    raise TypeError("Input contains a non-int or non-float")    
            else:
                raise TypeError("Input should be a list of int or float")


    def __repr__(self):
        name = type(self).__name__
        return f"{name} has a R^{self.dim_x} to R^{self.dim_f} function"


    def __str__(self):
        return self.__repr__()
        

    def grad(self, x_vec):
        """Evaluate the gradient of function at x_vec

        Input Arguments:
        =================
        x_vec: a scalar (float/int/np.number) or a list of scalars

        Return Arguments:
        =================
        gradient of the function at x_vec in np.array format with shape (dim_f, dim_x)
        EXCEPT when the the function is 1 to 1 mapping, 
        i.e. f: R -> R, f: R -> [R], f: R -> np.array([R])
        then, the output will mimic your function output format

        Examples
        =================
        >>> f = AutoDiff(lambda x: x**2)
        >>> f.grad(1)
        2.0
        >>> f = AutoDiff(lambda x: np.array([x**2]))
        >>> f.grad(1)
        array([2.])
        >>> f = AutoDiff(lambda x, y: sin(x*y))
        >>> f.grad([1,2])
        np.array([[-0.83229367, -0.41614684]])
        >>> f.grad([1,2]).shape
        (1, 2)
        >>> f = AutoDiff(lambda x, y: [sin(x*y), x+y])
        >>> f.grad([1,2])
        array([[-0.83229367, -0.41614684],
               [ 1.        ,  1.        ]])
        >>> f.grad([1,2]).shape 
        (2, 2)
        >>> f = AutoDiff(lambda x: [x, x**2])
        >>> f.grad(1)
        array([[1.],
               [2.]])
        >>> f.grad(1).shape
        (2, 1)
        """
        if self.dim_x == 1:
            # in case user input a length-1 list/np.array
            if isinstance(x_vec, (list, np.ndarray)):
                if len(x_vec) != 1:
                    raise ValueError("Your input dimension does not match the function input dimension")
                if isinstance(x_vec[0], (int, float, np.number)):
                    x_vec = float(x_vec[0])
                else:
                    raise TypeError("Input contains a non-int or non-float")  
            if isinstance(x_vec, (int, float, np.number)):
                x_vec = float(x_vec)
                if self.dim_f == 1:
                    output_1step_before = self.function(DualNumber(x_vec))
                    # if user defined function returns a scalar
                    if isinstance(output_1step_before, DualNumber):
                        return self.function(DualNumber(x_vec)).dual
                    # if user defined function returns a list
                    elif isinstance(output_1step_before, list):
                        return [self.function(DualNumber(x_vec))[0].dual]
                    # if user defined function returns a np.array
                    elif isinstance(output_1step_before, np.ndarray):
                        return np.array([self.function(DualNumber(x_vec))[0].dual])   
                else:
                    # one var func with multiple outputs f(x) = [f_0(x), f_1(x), ...]
                    # output components are listed as rows of the Jacobian matrix
                    # i.e. np.array[[df_0/dx], [df_1/dx], ...]
                    return np.transpose(np.array([[df_j_dx.dual for df_j_dx in self.function(DualNumber(x_vec))]]))
            else:
                raise TypeError("Input should be a scalar (int or float)")
        else:
            if len(x_vec) != self.dim_x:
                raise ValueError("Input dimension should match the function input dimension")
            if isinstance(x_vec, (list, np.ndarray)):
                if all(isinstance(x_i, (int, float, np.number)) for x_i in x_vec):
                    x_vec = [float(x_i) for x_i in list(x_vec)]
                    Jacobian = []
                    for i in range(self.dim_x):
                        x_vec_4grad = deepcopy(x_vec)
                        # apply d/dx_i and thus convert the i-th component of x_vec to DualNumber
                        x_vec_4grad[i] = DualNumber(x_vec[i]) 
                        df_dx_i = self.function(*x_vec_4grad)
                        if self.dim_f == 1:
                            # return a list of gradients i.e. [d/dx_0, d/dx_1, ...]f(x_vec)
                            Jacobian.append([df_dx_i.dual]) 
                        else:
                            # return a list of gradients i.e. [d/dx_0, d/dx_1, ...][f_0(x_vec), f_1(x_vec), ...]
                            Jacobian.append([df_dx_i[j].dual if isinstance(df_dx_i[j], DualNumber) else 0 for j in range(self.dim_f)]) 
                    return np.array(Jacobian).transpose()
                else:
                    raise TypeError("Input contains a non-int or non-float")
                    
            else:
                raise TypeError("Input should be a list of int or float")
