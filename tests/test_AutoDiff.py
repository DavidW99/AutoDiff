import sys
sys.path.append('../')
from AutoDiff.AutoDiff import AutoDiff as AD
from AutoDiff.AutoDiff import exam_user_function
from AutoDiff.ElementFunction import *
from AutoDiff.DualNumber import DualNumber
import numpy as np
import pdb
import pytest
import types

class TestAutoDiff:
    """Test class for AutoDiff types"""

    def test_exam_user_function(self):
        """test your the exam_user_function is correctly implemented"""
        def f_no_input():
            return 1
        def f_no_output(x):
            pass
        ## check non-function type
        with pytest.raises(TypeError):
            exam_user_function("p")
        ## check no output
        with pytest.raises(TypeError):
            exam_user_function(f_no_output)
        ## check no input
        with pytest.raises(TypeError):
            exam_user_function(f_no_input)
        ## check multi-dim output
        with pytest.raises(TypeError):
            exam_user_function(lambda x: [[x]])
        ## check 0-d array output
        with pytest.raises(TypeError):
            exam_user_function(lambda x: np.array(x))
        ## check 1-d array output containing non-numeric elements
        with pytest.raises(TypeError):
            exam_user_function(lambda x: np.array([x, "p"]))
        ## check output of empty list
        with pytest.raises(TypeError):
            exam_user_function(lambda x: [])
        ## check array output
        assert exam_user_function(lambda x: np.array([x])) == True
        ## check list output
        assert exam_user_function(lambda x: [x]) == True
        ## check int output
        assert exam_user_function(lambda x: x) == True
        ## check float output
        assert exam_user_function(lambda x: float(x)) == True
        ## check np.number output
        assert exam_user_function(lambda x: np.array([x])[0]) == True

    def test_init(self):
        """test if input is NOT of function type"""
        with pytest.raises(TypeError):
            AD([6,6,6])
        """test 1 to 1 mapping"""
        f = AD(lambda x: sin(x))
        assert f.dim_x==1, "AutoDiff is not initialized with correct input dimension."
        assert f.dim_f==1, "AutoDiff is not initialized with correct output dimension."
        """test 1 to n mapping"""
        f = AD(lambda x: [sin(x), cos(x), x**2])
        assert f.dim_x==1, "AutoDiff is not initialized with correct input dimension."
        assert f.dim_f==3, "AutoDiff is not initialized with correct output dimension."
        """test n to 1 mapping"""
        f = AD(lambda x, y: sin(x*y))
        assert f.dim_x==2, "AutoDiff is not initialized with correct input dimension."
        assert f.dim_f==1, "AutoDiff is not initialized with correct output dimension."
        """test n to m mapping"""
        f = AD(lambda x, y: [sin(x*y), cos(x*y), x**2])
        assert f.dim_x==2, "AutoDiff is not initialized with correct input dimension."
        assert f.dim_f==3, "AutoDiff is not initialized with correct output dimension."
        """test __str__ method"""
        f = AD(lambda x, y: [sin(x*y), cos(x*y), x**2])
        assert f.__str__() == "AutoDiff has a R^2 to R^3 function", "AutoDiff.__str__() is not implemented correctly."
        """test __repr__ method"""
        f = AD(lambda x, y: [sin(x*y), cos(x*y)])
        assert f.__repr__() == "AutoDiff has a R^2 to R^2 function", "AutoDiff.__repr__() is not implemented correctly."


    def test_call_1_to_1(self):
        """test 1 to 1 mapping"""
        f = AD(lambda x: sin(x))
        ## error cases
        # test non-numeric input
        x_vec = "p"
        with pytest.raises(TypeError):
            f(x_vec)
        # test input containing non-numeric elements
        x_vec = [DualNumber(1,1)]
        with pytest.raises(TypeError):
            f(x_vec)
        # test incorrect input dimension
        x_vec = [1,2]
        with pytest.raises(ValueError):
            f(x_vec)
        ## correct cases
        # input is of int or float type
        x_vec = 10.0
        assert np.allclose(f(x_vec), -0.54402111088), "Incorrect 1 to 1 mapping function evaluation for float input."
        # input is of list or np.array type
        x_vec = [10.0]
        assert np.allclose(f(x_vec), -0.54402111088), "Incorrect 1 to 1 mapping function evaluation for list input."
        x_vec = np.array([10.0])
        assert np.allclose(f(x_vec), -0.54402111088), "Incorrect 1 to 1 mapping function evaluation for np.array input."


    def test_call_1_to_n(self):
        """test 1 to n mapping"""
        f = AD(lambda x: [sin(x), cos(x), x**2])
        # test incorrect input dimension
        x_vec = [1,2,'p']   
        with pytest.raises(ValueError):
            f(x_vec)
        # input is of int or float type
        x_vec = 10.0
        assert np.allclose(f(x_vec), [-0.54402111088, -0.83907152906, 100.0]), "Incorrect 1 to n mapping function evaluation for float input."
        # input is of list or np.array type
        x_vec = [10.0]
        assert np.allclose(f(x_vec), [-0.54402111088, -0.83907152906, 100.0]), "Incorrect 1 to n mapping function evaluation for list input."
        x_vec = np.array([10.0])
        assert np.allclose(f(x_vec), [-0.54402111088, -0.83907152906, 100.0]), "Incorrect 1 to n mapping function evaluation for np.array input."


    def test_call_n_to_1(self):
        """test n to 1 mapping"""
        f = AD(lambda x, y: sin(x*y))
        # wrong input dimension
        x_vec = 10.0
        with pytest.raises(ValueError):
            f(x_vec)
        x_vec = 10.0, 2.0
        with pytest.raises(TypeError):
            f(x_vec)
        x_vec = [10.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            f(x_vec)
        # input contains non-numeric elements
        x_vec = np.array([10.0, 's'])
        with pytest.raises(TypeError):
            f(x_vec)
        # wrong input type which has len()
        x_vec = {'a':1, 'b':3}
        with pytest.raises(TypeError):
            f(x_vec)
        #input is of list or np.array type
        x_vec = [10.0, 2.0]
        assert np.allclose(f(x_vec), 0.9129452507), "Incorrect n to 1 mapping function evaluation for list input."
        x_vec = np.array([10.0, 2.0])
        assert np.allclose(f(x_vec), 0.9129452507), "Incorrect n to 1 mapping function evaluation for np.array input."
        # input contains a non-numeric element
        x_vec = np.array([10.0,[2.0]], dtype=object)
        with pytest.raises(TypeError):
            f(x_vec)


    def test_call_n_to_m(self):
        """test n to m mapping"""
        # test array output
        f = AD(lambda x, y: np.array([sin(x*y), cos(x*y), x**2]))
        x_vec = [10.0, 2.0]
        assert np.allclose(f(x_vec), [0.9129452507, 0.4080820618, 100.0]), "Incorrect n to m mapping function evaluation for list input."

    def test_call_vw(self):
        """==========================================================="""
        """1.Test if function input dimension =1"""
        """A.dim_x=1, dim_f=1"""
        f1A = lambda x: sin(x)
        AD1A = AD(f1A)
        """a.function input is of int or float type"""
        x_vec1Aa = [10.0] 
        assert np.allclose(AD1A(x_vec1Aa), -0.54402111088),"Incorrect function evaluation at 1-D input." 
        """b.function input is NOT of int or float type"""
        x_vec1Ab = np.array(["Input is a string"])
        try:
            AD1A(x_vec1Ab)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised

        """---------------------------------------------------"""
        """B.dim_x=1, dim_f>1"""
        f1B = lambda x: [cos(1/x),tan(2*x)]
        AD1B = AD(f1B)
        """a.""function input is of int or float type"""
        x_vec1Ba = np.array([20])
        #update this when david 
        assert np.allclose(AD1B(x_vec1Ba), [0.99875026039,-1.11721493092]),"Incorrect function evaluation at 1-D input."
        """b.function input is NOT of int or float type"""
        x_vec1Bb = "Input is a string"
        try:
            AD1B(x_vec1Bb)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """==============================================================="""
        """2.Test if function input dimension >1"""
        """A.dim_x<dim_f"""
        f2A = lambda x,y,z: [log10(1/x),arcsin(1/y),x**y,z]
        AD2A = AD(f2A)
        """a.Function Input dimension mismatch"""
        x_vec2Aa = [1,0.55]
        try:
            AD2A(x_vec2Aa)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised        
        """b.Function Input not of List or Array type"""
        x_vec2Ab = (3,5,9)
        try:
            AD2A(x_vec2Ab)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised     
        """c.Function Input contains non-int or non-float type"""
        x_vec2Ac = np.array([1,3,"include a string"])
        try:
            AD2A(x_vec2Ac)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised 
        """d.Fnction Input valid"""
        x_vec2Ad = [5,5,3]
        assert np.allclose(AD2A(x_vec2Ad), [-0.69897000,0.201357921,3125,3.]),"Incorrect function evaluation with input dimension>1."
        print(AD2A(x_vec2Ad))
        """---------------------------------------------------"""
        """B.dim_x>dim_f"""
        f2B = lambda x,y,z: np.array([log10(1/x),arcsin(1/y)])
        AD2B = AD(f2B)
        """a.Function Input dimension mismatch"""
        x_vec2Ba = np.array([1,0.55,0,30002])
        try:
            AD2B(x_vec2Ba)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised        
        """b.Function Input not of List or Array type"""
        x_vec2Bb = (3,2,3)
        try:
            AD2B(x_vec2Bb)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised     
        """c.Function Input contains non-int or non-float type"""
        x_vec2Bc = np.array([1,"inputstring",3])
        try:
            AD2B(x_vec2Bc)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised 
        """d.Fnction Input valid"""
        x_vec2Bd = [5,5,3]
        assert np.allclose(AD2B(x_vec2Bd), np.array([[-0.69897000,0.201357921]])),"Incorrect function evaluation with input dimension>1."
        """==============================================================="""


    def test_grad_1_to_1(self):
        """ test gradient of a 1-to-1 function """
        # scalar input with scalar output
        f = AD(lambda x: x**2)
        x_vec = 10.0
        assert np.allclose(f.grad(x_vec), 20), "incorrect gradient evaluation at 1-D input."
        # list/array input with scalar output
        f = AD(lambda x: x**2)
        x_vec = np.array([10.0])
        assert np.allclose(f.grad(x_vec), 20), "incorrect gradient evaluation at 1-D input."
        # test np.number input
        assert np.allclose(f.grad(x_vec[0]), 20), "incorrect gradient evaluation at 1-D input."
        x_vec = [10.0]
        assert np.allclose(f.grad(x_vec), 20), "incorrect gradient evaluation at 1-D input."
        # scalar input with list/array output
        f = AD(lambda x: [x**2])
        x_vec = 10.0
        assert f.grad(x_vec) == [20.0], "incorrect gradient evaluation at 1-D input."
        f = AD(lambda x: np.array([x**2]))
        x_vec = 1.0
        assert f.grad(x_vec) == np.array([2]), "incorrect gradient evaluation at 1-D input."


    def test_grad_1_to_n(self):
        """ test gradient of a 1-to-n function """
        # scalar input with list/array output
        f = AD(lambda x: np.array([x**2, x**3]))
        x_vec = 1.0
        assert np.allclose(f.grad(x_vec), np.array([[2.],[3.]])), "incorrect gradient evaluation at 1-D input."
        # list/array input with list/array output
        x_vec = np.array([2.0])
        assert np.allclose(f.grad(x_vec), np.array([[4.],[12.]])), "incorrect gradient evaluation at 1-D input."
        # wrong input type
        with pytest.raises(TypeError):
            f.grad("string")
        # wrong input dimension
        with pytest.raises(ValueError):
            f.grad(np.array([1.0, 2.0]))
        # wrong input dimension with wrong input type
        with pytest.raises(TypeError):
            f.grad(["string"])

    def test_grad_n_to_1(self):
        """ test gradient of a n-to-1 function """
        # list/array input with scalar output
        f = AD(lambda x, y: x**2 + sin(y))
        x_vec = np.array([1.0, 2.0])
        assert np.allclose(f.grad(x_vec), np.array([2., cos(2.0)])), "incorrect gradient evaluation with input dimension>1."
        # wrong input type
        with pytest.raises(ValueError):
            f.grad("string")
        # wrong input dimension
        with pytest.raises(ValueError):
            f.grad(np.array([1.0]))
        # wrong input dimension with wrong input type
        with pytest.raises(ValueError):
            f.grad(["string"])


    def test_grad_n_to_m_vw(self):
        """==========================================================="""
        """1.Test if function input dimension=1"""
        """A.dim_x=1, dim_f=1"""
        AD1A = AD(lambda x: x**2)
        """a.function input is of int or float type"""
        x_vec1Aa = 10.0
        assert np.allclose(AD1A.grad(x_vec1Aa), 20),"incorrect gradient evaluation at 1-D input." 
        """b.function input is NOT of int or float type"""
        x_vec1Ab = np.array(["Input is a string"]) 
        try:
            AD1A.grad(x_vec1Ab)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised

        """---------------------------------------------------"""
        """B.dim_x=1, dim_f>1"""
        f1B = lambda x: [sin(1/x),tan(2*x)]
        AD1B = AD(f1B)
        """a.""function input is of int or float type"""
        x_vec1Ba = np.array([20])
        assert np.allclose(AD1B.grad(x_vec1Ba), np.transpose(np.array([[-0.00249687565,4.49633840376]]))),"Incorrect function evaluation at 1-D input."
        """b.function input is NOT of int or float type"""
        x_vec1Bb = "Input is a string"
        try:
            AD1B.grad(x_vec1Bb)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """==============================================================="""
        """2.Test if function input dimension >1"""
        """A.dim_x<dim_f"""
        f2A = lambda x,y,z: [log10(1/x),sin(x),x**y,z]
        AD2A = AD(f2A)
        """a.Function Input dimension mismatch"""
        x_vec2Aa = [1,0.55]
        try:
            AD2A.grad(x_vec2Aa)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised        
        """b.Function Input not of List or Array type"""
        x_vec2Ab = (3,5,9)
        try:
            AD2A.grad(x_vec2Ab)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised     
        """c.Function Input contains non-int or non-float type"""
        x_vec2Ac = np.array([1,3,"include a string"])
        try:
            AD2A.grad(x_vec2Ac)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised 
        """d.Fnction Input valid"""
        x_vec2Ad = [5,5,3]
        assert np.allclose(AD2A.grad(x_vec2Ad), np.array([-0.08685889638, 0, 0,  0.28366218546, 0,0, 3125.0, 5029.493476356563, 0, 0, 0, 1]).reshape(4,3)),"Incorrect function evaluation with input dimension>1."
        """---------------------------------------------------"""
        """B.dim_x>dim_f"""
        f2B = lambda x,y,z: [log10(1/x),sin(x)]
        AD2B = AD(f2B)
        """a.Function Input dimension mismatch"""
        x_vec2Ba = np.array([1,0.55,0,30002])
        try:
            AD2B.grad(x_vec2Ba)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised        
        """b.Function Input not of List or Array type"""
        x_vec2Bb = (3,2,3)
        try:
            AD2B.grad(x_vec2Bb)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised     
        """c.Function Input contains non-int or non-float type"""
        x_vec2Bc = np.array([1,"inputstring",3])
        try:
            AD2B.grad(x_vec2Bc)
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised 
        """d.Fnction Input valid"""
        x_vec2Bd = [5,5,3]
        assert np.allclose(AD2B.grad(x_vec2Bd), np.array([-0.08685889638, 0, 0,0.28366218546, 0,0]).reshape(2,3)),"Incorrect function evaluation with input dimension>1."
        """==============================================================="""
        
        
