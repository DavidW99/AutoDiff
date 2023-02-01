import sys
sys.path.append('../')
from AutoDiff.ElementFunction import *
from AutoDiff.DualNumber import DualNumber
import pytest
import numpy as np


class TestElementFunction:
    """Test class for ElementFunction"""

    def test_sin(self):
        x = DualNumber(1, 2)
        y = sin(x)
        assert np.allclose(y.real, np.sin(1)) and np.allclose(y.dual, np.cos(1) * 2), "Function sin() failed"
        x2 = DualNumber(np.pi/2)
        y2 = sin(x2)
        assert np.allclose(y2.real, 1) and np.allclose(y2.dual, 0 * 1), "Function sin() failed"
        x3 = 3.5
        y3 = sin(x3)
        assert y3 == np.sin(3.5), "Function sin() with floating input failed"       
        with pytest.raises(TypeError):
            sin("p")
            
    def test_sin_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = sin(x_vec)
        assert np.allclose(y_vec[0].real, np.sin(1)) and np.allclose(y_vec[0].dual, np.cos(1) * 2) and np.allclose(y_vec[1].real, np.sin(3)) and np.allclose(y_vec[1].dual, np.cos(3) * 2), "Function sin() with vector input failed"
        x3 = [1.5,3.5]
        y3 = sin(x3)
        assert y3[0] == np.sin(1.5) and y3[1] == np.sin(3.5), "Function sin() with floating vector input failed"       
        with pytest.raises(TypeError):
            sin([1,"p",3.5])
    
    def test_arcsin(self):
        x = DualNumber(0.5, 2)
        y = arcsin(x)
        assert np.allclose(y.real, np.arcsin(0.5)) and np.allclose(y.dual, 1/np.sqrt(1-0.5**2)*2), "Function arcsin() failed"
        x2 = DualNumber(0)
        y2 = arcsin(x2)
        assert np.allclose(y2.real, 0) and np.allclose(y2.dual, 1), "Function arcsin() failed"
        x3 = -0.5
        y3 = arcsin(x3)
        assert y3 == np.arcsin(-0.5), "Function arcsin() with floating input failed"
        with pytest.raises(TypeError):
            arcsin("p")
    
    def test_arcsin_with_vector(self):
        x_vec = [DualNumber(0,2), DualNumber(0.5,2)]
        y_vec = arcsin(x_vec)
        assert np.allclose(y_vec[0].real, 0) and np.allclose(y_vec[0].dual, 2) and np.allclose(y_vec[1].real, np.arcsin(0.5)) and np.allclose(y_vec[1].dual, 1/np.sqrt(1-0.5**2)*2), "Function arcsin() with vector input failed"
        x3 = [0.5, -0.5]
        y3 = arcsin(x3)
        assert y3[0] == np.arcsin(0.5) and y3[1] == np.arcsin(-0.5), "Function arcsin() with floating vector input failed"
        with pytest.raises(TypeError):
            arcsin([0,"p",0.5])

    def test_cos(self):
        x = DualNumber(1, 2)
        y = cos(x)
        assert np.allclose(y.real, np.cos(1)) and np.allclose(y.dual, -np.sin(1) * 2), "Function cos() failed"
        x2 = DualNumber(np.pi/2)
        y2 = cos(x2)
        assert np.allclose(y2.real, 0) and np.allclose(y2.dual, -1 * 1), "Function cos() failed"
        x3 = 3.5
        y3 = cos(x3)
        assert y3 == np.cos(3.5), "Function cos() with floating input failed"  
        with pytest.raises(TypeError):
            cos("p")
            
    def test_cos_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = cos(x_vec)
        assert np.allclose(y_vec[0].real, np.cos(1)) and np.allclose(y_vec[0].dual, -np.sin(1) * 2) and np.allclose(y_vec[1].real, np.cos(3)) and np.allclose(y_vec[1].dual, -np.sin(3) * 2), "Function cos() with vector input failed"
        x3 = [1.5,3.5]
        y3 = cos(x3)
        assert y3[0] == np.cos(1.5) and y3[1] == np.cos(3.5), "Function cos() with floating vector input failed"              
        with pytest.raises(TypeError):
            cos([1,"p",3.5])
    
    def test_arccos(self):
        x = DualNumber(0.5, 2)
        y = arccos(x)
        assert np.allclose(y.real, np.arccos(0.5)) and np.allclose(y.dual, -1/np.sqrt(1-0.5**2)*2), "Function arccos() failed"
        x2 = DualNumber(0.5)
        y2 = arccos(x2)
        assert np.allclose(y2.real, np.arccos(0.5)) and np.allclose(y2.dual, -1/np.sqrt(1-0.5**2)), "Function arccos() failed"
        x3 = -0.5
        y3 = arccos(x3)
        assert y3 == np.arccos(-0.5), "Function arccos() with floating input failed"
        with pytest.raises(TypeError):
            arccos("p")
    
    def test_arccos_with_vector(self):
        x_vec = [DualNumber(0,2), DualNumber(0.5,2)]
        y_vec = arccos(x_vec)
        assert np.allclose(y_vec[0].real, np.arccos(0)) and np.allclose(y_vec[0].dual, -1/np.sqrt(1-0)*2) and np.allclose(y_vec[1].real, np.arccos(0.5)) and np.allclose(y_vec[1].dual, -1/np.sqrt(1-0.5**2)*2), "Function arccos() with vector input failed"
        x3 = [0.5, -0.5]
        y3 = arccos(x3)
        assert y3[0] == np.arccos(0.5) and y3[1] == np.arccos(-0.5), "Function arccos() with floating vector input failed"
        with pytest.raises(TypeError):
            arccos([0,"p",0.5])
            
    def test_tan(self):
        x = DualNumber(1, 2)
        y = tan(x)
        assert np.allclose(y.real, np.tan(1)) and np.allclose(y.dual, 1/np.cos(1)**2 * 2), "Function tan() failed"
        x2 = DualNumber(2.5)
        y2 = tan(x2)
        assert np.allclose(y2.real, np.tan(2.5)) and np.allclose(y2.dual, 1/np.cos(2.5)**2 * 1), "Function tan() failed"
        x3 = 3.5
        y3 = tan(x3)
        assert y3 == np.tan(3.5), "Function tan() with floating input failed" 
        x4 = 3/2 *np.pi
        try:
            tan(x4)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised  
        x5 = DualNumber(3/2 *np.pi)
        try:
            tan(x5)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised          
        with pytest.raises(TypeError):
            tan("p")
            
    def test_tan_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = tan(x_vec)
        assert np.allclose(y_vec[0].real, np.tan(1)) and np.allclose(y_vec[0].dual, 1/np.cos(1)**2 * 2), "Function tan() with vector input failed"
        assert np.allclose(y_vec[1].real, np.tan(3)) and np.allclose(y_vec[1].dual, 1/np.cos(3)**2 * 2), "Function tan() with vector input failed"
        x3 = [1.5,3.5]
        y3 = tan(x3)
        assert y3[0] == np.tan(1.5) and y3[1] == np.tan(3.5), "Function tan() with floating vector input failed" 
        x4 = [24,3/2 *np.pi]
        try:
            tan(x4)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised       
        x5 = [24,DualNumber(3/2 *np.pi,2)]
        try:
            tan(x5)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised           
        with pytest.raises(TypeError):
            tan([1,"p",3.5])
    
    def test_arctan(self):
        x = DualNumber(5, 2)
        y = arctan(x)
        assert np.allclose(y.real, np.arctan(5)) and np.allclose(y.dual, 1/(1+5**2)*2), "Function arctan() failed"
        x2 = DualNumber(0)
        y2 = arctan(x2)
        assert np.allclose(y2.real, 0) and np.allclose(y2.dual, 1), "Function arctan() failed"
        x3 = -0.5
        y3 = arctan(x3)
        assert np.allclose(y3,np.arctan(-0.5)), "Function arctan() with floating input failed"
        with pytest.raises(TypeError):
            arctan("p")
    
    def test_arctan_with_vector(self):
        x_vec = [DualNumber(0,2), DualNumber(5,2)]
        y_vec = arctan(x_vec)
        assert np.allclose(y_vec[0].real, 0) and np.allclose(y_vec[0].dual, 2) and np.allclose(y_vec[1].real, np.arctan(5)) and np.allclose(y_vec[1].dual, 1/(1+5**2)*2), "Function arccos() with vector input failed"
        x3 = [0.5, -0.5]
        y3 = arctan(x3)
        assert y3[0] == np.arctan(0.5) and y3[1] == np.arctan(-0.5), "Function arccos() with floating vector input failed"
        with pytest.raises(TypeError):
            arctan([0,"p",0.5])

    def test_exp(self):
        x = DualNumber(1, 2)
        y = exp(x)
        assert np.allclose(y.real, np.exp(1)) and np.allclose(y.dual, np.exp(1) * 2), "Function exp() failed"
        x2 = DualNumber(2.5)
        y2 = exp(x2)
        assert np.allclose(y2.real, np.exp(2.5)) and np.allclose(y2.dual, np.exp(2.5) * 1), "Function exp() failed"
        x3 = 3.5
        y3 = exp(x3)
        assert y3 == np.exp(3.5), "Function exp() with floating input failed"   
        with pytest.raises(TypeError):
            exp("p")
            
    def test_exp_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = exp(x_vec)
        assert np.allclose(y_vec[0].real, np.exp(1)) and np.allclose(y_vec[0].dual, np.exp(1) * 2), "Function exp() with vector input failed"
        assert np.allclose(y_vec[1].real, np.exp(3)) and np.allclose(y_vec[1].dual, np.exp(3) * 2), "Function exp() with vector input failed"
        x3 = [1.5,3.5]
        y3 = exp(x3)
        assert y3[0] == np.exp(1.5) and y3[1] == np.exp(3.5), "Function exp() with floating vector input failed"              
        with pytest.raises(TypeError):
            exp([1,"p",3.5]) 

    def test_log(self):
        x = DualNumber(1,2)
        y = log(x)
        assert np.allclose(y.real, np.log(1)) and np.allclose(y.dual, 1/1 * 2), "Function log() failed"
        x2 = DualNumber(2.5)
        y2 = log(x2)
        assert np.allclose(y2.real, np.log(2.5)) and np.allclose(y2.dual, 1/2.5 * 1), "Function log() failed"
        x3 = 3.5
        y3 = log(x3)
        assert y3 == np.log(3.5), "Function log() with floating input failed" 
        x4 = -3
        try:
            log(x4)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised      
        x5 = DualNumber(-3.5,2)
        try:
            log(x5)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised       
        with pytest.raises(TypeError):
            log("p")

    def test_log_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = log(x_vec)
        assert np.allclose(y_vec[0].real, np.log(1)) and np.allclose(y_vec[0].dual, 1/1 * 2), "Function log() with vector input failed"
        assert np.allclose(y_vec[1].real, np.log(3)) and np.allclose(y_vec[1].dual, 1/3 * 2), "Function log() with vector input failed"
        x3 = [1.5,3.5]
        y3 = log(x3)
        assert y3[0] == np.log(1.5) and y3[1] == np.log(3.5), "Function log() with floating vector input failed" 
        x4 =  [-2.5,6]
        try:
            log(x4)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised     
        x5 =  [DualNumber(-2.5,2),6]
        try:
            log(x5)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised            
        with pytest.raises(TypeError):
            log([1,"p",3.5])
      
    #@pytest.mark.skip(reason="no way of currently testing this")
    # def test_power(self):
    #     x = DualNumber(1, 2)
    #     y = pow(x,4)
    #     assert np.allclose(y.real, 1 ** 4) and np.allclose(y.dual, 4 * (1**3) * 2), "Function pow() failed"
    #     x2 = DualNumber(2.5)
    #     y2 = pow(x2,4)
    #     assert np.allclose(y2.real, np.pow(2.5,4)) and np.allclose(y2.dual, 4 * 2.5 **3 * 1), "Function pow() failed"
    #     x3 = 3.5
    #     y3 = pow(x3)
    #     assert y3 == np.pow(3.5), "Function pow() with floating input failed"         
    #     with pytest.raises(TypeError):
    #         pow("p")

    # #@pytest.mark.skip(reason="no way of currently testing this")
    # def test_power_with_vector(self):
    #     x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
    #     y_vec = pow(x_vec,4)
    #     assert np.allclose(y_vec[0].real, 1 ** 4) and np.allclose(y_vec[0].dual, 4 * (1**3) * 2), "Function pow() with vector input failed"
    #     assert np.allclose(y_vec[1].real, 3 ** 4) and np.allclose(y_vec[1].dual, 4 * (3**3) * 2), "Function pow() with vector input failed"
    #     x3 = [1.5,3.5]
    #     y3 = pow(x3)
    #     assert y3[0] == np.pow(1.5) and y3[1] == np.pow(3.5), "Function pow() with floating vector input failed"              
    #     with pytest.raises(TypeError):
    #         pow([1,"p",3.5])   

    def test_sqrt(self):
        x = DualNumber(1, 2)
        y = sqrt(x)
        assert np.allclose(y.real, 1) and np.allclose(y.dual, 0.5 * 1**-0.5 * 2), "Function sqrt() failed"
        x2 = DualNumber(100)
        y2 = sqrt(x2)
        assert np.allclose(y2.real, 10) and np.allclose(y2.dual, 0.5 * 100 **-0.5), "Function sqrt() failed"
        x3 = 3.5
        y3 = sqrt(x3)
        assert y3 == np.sqrt(3.5), "Function sqrt() with floating input failed"
        x4 = -2.5
        try:
            sqrt(x4)
        except ValueError as e:
            print(e)
        x5 = DualNumber(-2.5,2)
        try:
            sqrt(x5)
        except ValueError as e:
            print(e)
        with pytest.raises(TypeError):
            sqrt("p")
            
    def test_sqrt_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = sqrt(x_vec)
        assert np.allclose(y_vec[0].real, 1) and np.allclose(y_vec[0].dual, 0.5 * 1 ** -0.5 * 2), "Function sqrt() with vector input failed"
        assert np.allclose(y_vec[1].real, np.sqrt(3)) and np.allclose(y_vec[1].dual, 0.5 * 3 ** -0.5 * 2), "Function sqrt() with vector input failed"
        x3 = [1.5,3.5]
        y3 = sqrt(x3)
        assert y3[0] == np.sqrt(1.5) and y3[1] == np.sqrt(3.5), "Function sqrt() with floating vector input failed"
        x4 = [-2.5,6]
        try:
            sqrt(x4)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised
        x5 = [DualNumber(-2.5,2),6]
        try:
            sqrt(x5)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised
        with pytest.raises(TypeError):
            sqrt([1,"p",3.5])
    
    def test_logBase(self):
        x = DualNumber(1,2)
        y = logBase(2,x) # base 2
        assert np.allclose(y.real, np.log(1)/np.log(2)) and np.allclose(y.dual, 2/(1*np.log(2))), "Function logBase() failed"
        x2 = DualNumber(2.5)
        y2 = logBase(3,x2)
        assert np.allclose(y2.real, np.log(2.5)/np.log(3)) and np.allclose(y2.dual, 1/(2.5*np.log(3))), "Function logBase() failed"
        x3 = 3.5
        y3 = logBase(2,x3)
        assert y3 == np.log(3.5)/np.log(2), "Function logBase() with floating input failed"
        x4 = -3
        try:
            logBase(2,x4)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised
        x5 = DualNumber(-3.5,2)
        try:
            logBase(2,x5)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised
        with pytest.raises(TypeError):
            logBase(2,"p")
            
    def test_logBase_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = logBase(2,x_vec) # base 2
        assert np.allclose(y_vec[0].real, 0) and np.allclose(y_vec[0].dual, 2/(1*np.log(2))), "Function logBase() with vector input failed"
        assert np.allclose(y_vec[1].real, np.log(3)/np.log(2)) and np.allclose(y_vec[1].dual, 2/(3*np.log(2))), "Function logBase() with vector input failed"
        x3 = [1.5,3.5]
        y3 = logBase(2,x3)
        assert y3[0] == np.log(1.5)/np.log(2) and y3[1] == np.log(3.5)/np.log(2), "Function logBase() with floating vector input failed"
        x4 = [-2.5,6]
        try:
            logBase(2,x4)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised
        x5 = [DualNumber(-2.5,2),6]
        try:
            logBase(2,x5)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised
        with pytest.raises(TypeError):
            logBase(2,[1,"p",3.5])
        with pytest.raises(AssertionError):
            logBase(-2,[1,3,5])
    
    def test_log10(self):
        x = DualNumber(1,2)
        y = log10(x)
        assert np.allclose(y.real, np.log10(1)) and np.allclose(y.dual, 2/(1*np.log(10))), "Function log10() failed"
        x2 = DualNumber(2.5)
        y2 = log10(x2)
        assert np.allclose(y2.real, np.log10(2.5)) and np.allclose(y2.dual, 1/(2.5*np.log(10))), "Function log10() failed"
        x3 = 3.5
        y3 = log10(x3)
        assert np.allclose(y3,np.log10(3.5)), "Function log10() with floating input failed"
        x4 = -3
        try:
            log10(x4)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised
        x5 = DualNumber(-3.5,2)
        try:
            log10(x5)
        except ValueError as e:
            print(e)  # it is expected that the above assertion is raised
        with pytest.raises(TypeError):
            log10("p")
    
    def test_log10_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = log10(x_vec)
        assert np.allclose(y_vec[0].real, 0) and np.allclose(y_vec[0].dual, 2/(1*np.log(10))), "Function log10() with vector input failed"
        assert np.allclose(y_vec[1].real, np.log10(3)) and np.allclose(y_vec[1].dual, 2/(3*np.log(10))), "Function log10() with vector input failed"
        x3 = [1.5,3.5]
        y3 = log10(x3)
        assert np.allclose(y3[0],np.log10(1.5)) and np.allclose(y3[1], np.log10(3.5)), "Function logBase() with floating vector input failed"
        x4 = [-2.5,6]
        try:
            log10(x4)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised
        x5 = [DualNumber(-2.5,2),6]
        try:
            log10(x5)
        except ValueError as e:
            print(e) # it is expected that the above assertion is raised
        with pytest.raises(TypeError):
            log10([1,"p",3.5])
    
    def test_sinh(self):
        x = DualNumber(1, 2)
        y = sinh(x)
        assert np.allclose(y.real, np.sinh(1)) and np.allclose(y.dual, np.cosh(1) * 2), "Function sinh() failed"
        x2 = DualNumber(np.pi/2)
        y2 = sinh(x2)
        assert np.allclose(y2.real, np.sinh(np.pi/2)) and np.allclose(y2.dual, np.cosh(np.pi/2) * 1), "Function sinh() failed"
        x3 = 3.5
        y3 = sinh(x3)
        assert np.allclose(y3,np.sinh(3.5)), "Function sinh() with floating input failed"
        with pytest.raises(TypeError):
            sinh("p")
    
    def test_sinh_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = sinh(x_vec)
        assert np.allclose(y_vec[0].real, np.sinh(1)) and np.allclose(y_vec[0].dual, np.cosh(1) * 2) and np.allclose(y_vec[1].real, np.sinh(3)) and np.allclose(y_vec[1].dual, np.cosh(3) * 2), "Function sinh() with vector input failed"
        x3 = [1.5,3.5]
        y3 = sinh(x3)
        assert np.allclose(y3[0], np.sinh(1.5)) and np.allclose(y3[1], np.sinh(3.5)), "Function sinh() with floating vector input failed"
        with pytest.raises(TypeError):
            sinh([1,"p",3.5])
    
    def test_cosh(self):
        x = DualNumber(1, 2)
        y = cosh(x)
        assert np.allclose(y.real, np.cosh(1)) and np.allclose(y.dual, np.sinh(1) * 2), "Function cosh() failed"
        x2 = DualNumber(np.pi/2)
        y2 = cosh(x2)
        assert np.allclose(y2.real, np.cosh(np.pi/2)) and np.allclose(y2.dual, np.sinh(np.pi/2) * 1), "Function cosh() failed"
        x3 = 3.5
        y3 = cosh(x3)
        assert np.allclose(y3,np.cosh(3.5)), "Function cosh() with floating input failed"
        with pytest.raises(TypeError):
            cosh("p")
    
    def test_cosh_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = cosh(x_vec)
        assert np.allclose(y_vec[0].real, np.cosh(1)) and np.allclose(y_vec[0].dual, np.sinh(1) * 2) and np.allclose(y_vec[1].real, np.cosh(3)) and np.allclose(y_vec[1].dual, np.sinh(3) * 2), "Function cosh() with vector input failed"
        x3 = [1.5,3.5]
        y3 = cosh(x3)
        assert np.allclose(y3[0], np.cosh(1.5)) and np.allclose(y3[1], np.cosh(3.5)), "Function cosh() with floating vector input failed"
        with pytest.raises(TypeError):
            cosh([1,"p",3.5])
    
    def test_tanh(self):
        x = DualNumber(1, 2)
        y = tanh(x)
        assert np.allclose(y.real, np.tanh(1)) and np.allclose(y.dual, (np.cosh(1)**2-np.sinh(1)**2)/np.cosh(1)**2 * 2), "Function tanh() failed"
        x2 = DualNumber(np.pi/2)
        y2 = tanh(x2)
        assert np.allclose(y2.real, np.tanh(np.pi/2)) and np.allclose(y2.dual, (np.cosh(np.pi/2)**2-np.sinh(np.pi/2)**2)/np.cosh(np.pi/2)**2), "Function tanh() failed"
        x3 = 3.5
        y3 = tanh(x3)
        assert np.allclose(y3,np.tanh(3.5)), "Function tanh() with floating input failed"
        with pytest.raises(TypeError):
            tanh("p")
            
    def test_tanh_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = tanh(x_vec)
        assert np.allclose(y_vec[0].real, np.tanh(1)) and np.allclose(y_vec[0].dual, (np.cosh(1)**2-np.sinh(1)**2)/np.cosh(1)**2 * 2) and np.allclose(y_vec[1].real, np.tanh(3)) and np.allclose(y_vec[1].dual, (np.cosh(3)**2-np.sinh(3)**2)/np.cosh(3)**2 * 2), "Function tanh() with vector input failed"
        x3 = [1.5,3.5]
        y3 = tanh(x3)
        assert np.allclose(y3[0], np.tanh(1.5)) and np.allclose(y3[1], np.tanh(3.5)), "Function tanh() with floating vector input failed"
        with pytest.raises(TypeError):
            tanh([1,"p",3.5])
    
    def test_logistic(self):
        x = DualNumber(1,2)
        y = logistic(x)
        assert np.allclose(y.real, 1/(1+np.exp(-1))) and np.allclose(y.dual, np.exp(-1)/(1+np.exp(-1))**2 *2), "Function logistic() failed"
        x2 = DualNumber(2.5)
        y2 = logistic(x2)
        assert np.allclose(y2.real, 1/(1+np.exp(-2.5))) and np.allclose(y2.dual, np.exp(-2.5)/(1+np.exp(-2.5))**2), "Function logistic() failed"
        x3 = 3.5
        y3 = logistic(x3)
        assert y3 == 1/(1+np.exp(-3.5)), "Function logistic() with floating input failed"
        with pytest.raises(TypeError):
            logistic("p")
            
    def test_logistic_with_vector(self):
        x_vec = [DualNumber(1, 2), DualNumber(3, 2)]
        y_vec = logistic(x_vec)
        assert np.allclose(y_vec[0].real, 1/(1+np.exp(-1))) and np.allclose(y_vec[0].dual, np.exp(-1)/(1+np.exp(-1))**2 *2), "Function logistic() with vector input failed"
        assert np.allclose(y_vec[1].real, 1/(1+np.exp(-3))) and np.allclose(y_vec[1].dual, np.exp(-3)/(1+np.exp(-3))**2 *2), "Function logistic() with vector input failed"
        x3 = [1.5,3.5]
        y3 = logistic(x3)
        assert y3[0] == 1/(1+np.exp(-1.5)) and y3[1] == 1/(1+np.exp(-3.5)), "Function logBase() with floating vector input failed"
        with pytest.raises(TypeError):
            logistic([1,"p",3.5])
    
