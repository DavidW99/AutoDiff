import sys
sys.path.append('../')
from AutoDiff.feature import *

import pytest
import numpy as np

class Test_grad_descent:
    """Test class for AutoDiff types"""

    def test_user_function_input(self):
        # test wrong output dimension
        f = lambda x: [(x-1)**2, (x-2)**2]
        with pytest.raises(TypeError):
            grad_descent(f, x0=1)
        # test wrong output type
        f = lambda x: [x]
        with pytest.raises(TypeError):
            grad_descent(f, x0=1)
        # test 1 to 1 function
        f = lambda x: x**2
        with pytest.raises(TypeError):
            grad_descent(f, x0=[1])
        # test wrong input type
        f = lambda x, y: (x-y)**2
        ## input is not an array for dim_x >1
        with pytest.raises(TypeError):
            grad_descent(f, x0=1)
        ## input is a multi-dim array
        with pytest.raises(TypeError):
            grad_descent(f, x0=np.array([[1,2]]))
        ## input array has non-numerical elements
        with pytest.raises(TypeError):
            grad_descent(f, x0=np.array([1,'string']))

    def test_grad_descent_algo(self):
        # test 1D quartic function
        f = lambda x: -10*x**2 + x**4 - x 
        x0 = 4
        x = grad_descent(f, x0, alpha=1e-5, verbose=False)
        assert np.isclose(x, 2.261, atol=1e-3), 'single-variable function gradient descent failed'
        # test plotting
        x = grad_descent(f, x0, alpha=1e-5, max_iter=2, verbose=True)
        # test maximum iteration
        x = grad_descent(f, x0, alpha=1e-5, max_iter=1, verbose=False)
        # test n-var function
        f = lambda x, y: sin((x-0.1)**2+(y)**2)
        x0 = np.array([0.5, 0.2])
        x = grad_descent(f, x0, alpha=1e-4)
        assert np.allclose(x, np.array([0.10, 0.0]), atol=1e-3), 'n-variable function gradient descent failed'
        # test n-var function verbose mode
        x = grad_descent(f, x0, max_iter=1, verbose=True)
        
    

