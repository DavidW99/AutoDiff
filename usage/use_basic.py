import sys
sys.path.append('../')

from AutoDiff.AutoDiff import AutoDiff as AD
from AutoDiff.AutoDiff import exam_user_function
from AutoDiff.ElementFunction import *
from AutoDiff.DualNumber import DualNumber
from AutoDiff.feature import grad_descent
import numpy as np
import pdb


### 1. Define your function
f1 = lambda x, y: np.array([sin(x*y), exp(x) + y, y**2]) 

### 2. Create an AutoDiff object
myAD = AD(f1)
print(myAD)

### 3. Create x_vec, where its 1st component refers to x, and 2nd component refers to y in f1(x,y)
x_vec = np.array([np.pi/2, 3])

### 4. Get the function value and gradient given the point of evaluation x_vec
print(myAD(x_vec))
print(f'The dimension of your Jacobian is {myAD.dim_f}x{myAD.dim_x}.\n')
print(myAD.grad(x_vec))
print("""the Jacobian matrix is expected to have the following form:

    [[y*cos(x*y), x*cos(x*y)],
    [exp(x)    , 1         ],
    [0         , 2*y       ]]

where row refers to the i-th component of function f1(x,y), and column refers to j-th component of variable x_vec""")
print("Here, let's print out the expected gradient value\n")
print(np.array([[3*np.cos(np.pi/2*3), np.pi/2*np.cos(np.pi/2*3)], [exp(np.pi/2), 1], [0, 2*3]]))

#pdb.set_trace()