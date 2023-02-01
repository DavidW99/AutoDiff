import sys
sys.path.append('../')
from AutoDiff.DualNumber import DualNumber
import pytest
import sys
import numpy as np


class TestDualNumber:
    """Test class for DualNumber types"""

    def test_init(self):
        """test if both real and dual parts are integer"""
        x1 = DualNumber(1, 2)
        assert x1.real == 1 and x1.dual == 2, "DualNumber is not initialized correctly"
        """test if both real part is float and dual part is integer"""
        x2 = DualNumber(3.5, 2)
        assert x2.real == 3.5 and x2.dual == 2, "DualNumber is not initialized correctly"
        """test if both real part is integer and dual part is float"""
        x3 = DualNumber(6, 5.5)
        assert x3.real == 6 and x3.dual == 5.5, "DualNumber is not initialized correctly"
        """test if both real and dual parts are float"""
        x4 = DualNumber(9.666, 100.345)
        assert x4.real == 9.666 and x4.dual == 100.345, "DualNumber is not initialized correctly"
        """test if real part is not a integer or float, or both parts are not valid type"""
        try:
            x5= DualNumber("testlol", [1,2,3,4])
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        try:
            x6 = DualNumber(5, ("not","working"))
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        try:
            x7=DualNumber({9:2,4:7}, [1,2,3,4])
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """test if the default dual value works as expected"""
        x8=DualNumber(0.00001)
        assert x8.real == 0.00001 and x8.dual == 1, "DualNumber is not initialized correctly"
        """test __str__"""
        x9=DualNumber(2.5,6)
        assert str(x9) == "2.5 + 6 eps", "__str__ method is not working properly"
        
        
    def test_addition(self):
        """test addition dual+real"""
        x1 = DualNumber(1, 5)
        x2 = x1 + 1
        x3 = x1 + 3.0
        assert x2.real == 2 and x2.dual == 5, "DualNumber __add__ for int scalar is not implemented correctly"
        assert x3.real == 4.0 and x3.dual == 5, "DualNumber __add__ for float scalar is not implemented correctly"
        """test revese addition"""
        x4 = 1 + x1
        x5 = 3.0 + x1
        assert x4.real == 2 and x4.dual == 5, "DualNumber __radd__ for two DualNumbers is not implemented correctly"
        assert x5.real == 4.0 and x5.dual == 5, "DualNumber __radd__ for float scalar is not implemented correctly"
        """test addition dual+dual"""
        x6 = x2 + x3
        x7 = x6 + DualNumber(10.2, 5.5)
        assert x6.real == 6.0 and x6.dual == 10, "DualNumber __add__ for dual is not implemented correctly"
        assert x7.real == 16.2 and x7.dual == 15.5, "DualNumber __add__ for dual is not implemented correctly"
        """test augmentation"""
        x8 = DualNumber(98, 66)
        x8 +=2
        assert x8.real == 100 and x8.dual == 66, "DualNumber __add__ for int scalar is not implemented correctly"
        x9 = DualNumber(98, 66)
        x9 +=DualNumber(10.2, 2.2)
        assert x9.real == 108.2 and x9.dual == 68.2, "DualNumber __add__ for dual is not implemented correctly"
        """test addition of invalid data type"""
        try:
            x1 + "test"
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """test revsere addition of invalid data type"""
        try:
           [8,9,888]+ x1 
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
       
    def test_substraction(self):
        """test dual-real"""
        x1 = DualNumber(4, 2)
        x2 = x1 - 1
        x3 = x1 - 3.0
        assert x2.real == 3 and x2.dual == 2, "DualNumber __sub__ for int scalar is not implemented correctly"
        assert x3.real == 1.0 and x3.dual == 2, "DualNumber __sub__ for float scalar is not implemented correctly"
        """test revese subtraction"""
        x4 = 1 - x1
        x5 = 3.0 - x1
        assert x4.real == -3 and x4.dual == -2, "DualNumber __rsub__ for int scalar is not implemented correctly"
        assert x5.real == -1.0 and x5.dual == -2, "DualNumber __rsub__ for float scalar is not implemented correctly"
        """test dual-dual"""
        x6 = x2 - x3
        x7 = x6 - DualNumber(10.2, 5.5)
        assert x6.real == 2.0 and x6.dual == 0, "DualNumber __sub__ for dual is not implemented correctly"
        assert x7.real == -8.2 and x7.dual == -5.5, "DualNumber __sub__ for dual is not implemented correctly"
        """test decrement"""
        x8 = DualNumber(98, 66)
        x8 -=2
        assert x8.real == 96 and x8.dual == 66, "DualNumber __sub__ for dual is not implemented correctly"
        x9 = DualNumber(98, 66)
        x9 -=DualNumber(17.5, 5)
        assert x9.real == 80.5 and x9.dual == 61, "DualNumber __sub__ for dual is not implemented correctly"
        """test subtraction of invalid data type"""
        try:
            x1 - "test"
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """test reverse subtraction of invalid data type"""
        try:
            [8,9,888]- x1 
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised

    def test_multiplication(self):
        """test dual*real"""
        x1 = DualNumber(3, 2)
        x2 = x1 * 1
        x3 = x1 * 3.0
        assert x2.real == 3 and x2.dual == 2, "DualNumber __mul__ for int scalar is not implemented correctly"
        assert x3.real == 9.0 and x3.dual == 6.0, "DualNumber __mul__ for float scalar is not implemented correctly"
        """test revese multiplication"""
        x4 = 1 * x1
        x5 = 3.0 * x1
        assert x4.real == 3 and x4.dual == 2, "DualNumber __rmul__ for int scalar is not implemented correctly"
        assert x5.real == 9.0 and x5.dual == 6.0, "DualNumber __rml__ for float scalar is not implemented correctly"
        """test addition dual*dual"""
        x6 = x2 * x3
        x7 = x6 * DualNumber(10, 20)
        assert x6.real == 27.0 and x6.dual == 36.0, "DualNumber __mul__ for dual is not implemented correctly"
        assert x7.real == 270.0 and x7.dual == 900.0, "DualNumber __mul__ for dual is not implemented correctly"
        """test augmentation"""
        x8 = DualNumber(98, 66)
        x8 *= 5
        assert x8.real == 490 and x8.dual == 330, "DualNumber __mul__ for dual is not implemented correctly"
        x9 = DualNumber(5.5, 5)
        x9 *=DualNumber(6, 10)
        assert x9.real == 33.0 and x9.dual == 85.0, "DualNumber __mul__ for dual is not implemented correctly"
        """test multiplication of invalid data type"""
        try:
            x1 * "test"
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """test reverse multiplication of invalid data type"""
        try:
           [8,9,888]* x1 
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
    

    def test_division(self):
        """test dual/real"""
        x1 = DualNumber(24, 300)
        x2 = x1 /4
        x3 = x1 / 3.0
        assert x2.real == 6 and x2.dual == 75, "DualNumber __truediv__ for int scalar is not implemented correctly"
        assert x3.real == 8.0 and x3.dual == 100.0, "DualNumber __truediv__ for float scalar is not implemented correctly"
        """test revese division"""
        x4 = 900 / x1
        x5 = 1800.0 / x1

        assert x4.real == 37.5 and x4.dual == -468.75, "DualNumber __rtruediv__ for int scalar is not implemented correctly"
        assert x5.real == 75.0 and x5.dual == -937.5, "DualNumber __rtruediv__ for float scalar is not implemented correctly"
        """test division dual/dual"""
        x6 = x2 / x3
        x7 = x1 / DualNumber(10, 20)
        assert x6.real == 0.75 and x6.dual == 0.0, "DualNumber  __truediv__ for dual is not implemented correctly"
        assert x7.real == 2.4 and x7.dual == 25.2, "DualNumber  __truediv__ for dual is not implemented correctly"
        """test augmentation"""
        x8 = DualNumber(98, 66)
        x8 /= 5
        assert x8.real == 19.6 and x8.dual == 13.2, "DualNumber __truediv__  for dual is not implemented correctly"
        x9 = DualNumber(5.5, 5)
        x9 /=DualNumber(6, 10)
        assert x9.real == (5.5/6) and x9.dual == (-25/36), "DualNumber __truediv__  for dual is not implemented correctly"
        """test division by zero"""
        try:
            x1/0
        except ZeroDivisionError as e:
            print(e)  # it is expected that the above assertion is raised
        """test division of invalid data type"""
        try:
            x1/"test"
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """test reverse division of invalid data type"""
        try:
           [8,9,888]/x1 
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
 
    def test_negation(self):
        x1 = DualNumber(1, 2)
        y1 = -x1
        assert y1.real == -1 and y1.dual == -2, "DualNumber __neg__ failed"
        x2 = DualNumber(-0.1)
        y2 = -x2
        assert y2.real == 0.1 and y2.dual == -1, "DualNumber __neg__ failed"

    def test_positive(self):
        x1 = DualNumber(1, 2)
        y1 = +x1
        assert y1.real == 1 and y1.dual == 2, "DualNumber __pos__ failed"
        x2 = DualNumber(0.1)
        y2 = +x2
        assert y2.real == 0.1 and y2.dual == 1, "DualNumber __pos__ failed"

    def test_power(self):
        """test dual**real"""
        x1 = DualNumber(24, 300)
        x2 = x1**4
        x3 = x1**3.1
        assert x2.real == 331776 and x2.dual == 16588800, "DualNumber __pow__ for int scalar is not implemented correctly"
        assert x3.real == 24**3.1 and x3.dual == (24**2.1)*930, "DualNumber __pow__ for float scalar is not implemented correctly"
        """test revese power"""
        x4 = 900**x1
        x5 = 1800.0**x1
        assert x4.real == 900**24 and x4.dual == (900**24)*300*np.log(900), "DualNumber __rpow__ for int scalar is not implemented correctly"
        assert x5.real == (1800**24)*1.0 and x5.dual == (1800**24)*300*np.log(1800), "DualNumber __rpow__ for int scalar is not implemented correctly"
        """test dual**dual"""
        x2 = DualNumber(5,2)
        x3 = DualNumber(3,7)
        x6 = x2 ** x3
        x7 = x1**DualNumber(10.5, 20.0)
        assert x6.real == 125 and x6.dual == 125*(7*np.log(5)+1.2), "DualNumber  __pow__ for dual is not implemented correctly"
        assert x7.real == 24**10.5 and x7.dual == (24**10.5)*(20*np.log(24)+131.25), "DualNumber  __pow__ for dual is not implemented correctly"
        """test augmentation"""
        x8 = DualNumber(98, 66)
        x8 **= 5
        assert x8.real == 98**5 and x8.dual == (98**4)*330, "DualNumber __pow__ for dual is not implemented correctly"
        x9 = DualNumber(5.5, 5)
        x9 **=DualNumber(6, 10)
        assert x9.real == 5.5**6 and x9.dual == (5.5**6)*(10*np.log(5.5)+(30/5.5)), "DualNumber __pow__for dual is not implemented correctly"
        """test power of invalid data type"""
        try:
            x1**"test"
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        """test reverse power of invalid data type"""
        try:
           [8,9,888]**x1 
        except TypeError as e:
            print(e)  # it is expected that the above assertion is raised
        ## TODO: more tests
        x = DualNumber(2, 3)
        y = x ** 2.0
        assert y.real == 4.0 and y.dual == 12.0, "DualNumber __pow__ with scalar exponent failed"

    def test_reflective_operators(self):
        ## TODO: more tests
        # Test reflective operators for the DualNumber type (what happens
        # when you do `1 + z` instead of `z + 1` where `z` is a complex number).
        """test oepration between real and dual"""
        x1 = DualNumber(3, 2)
        xl1 = 1 + x1 
        xr1 = x1 + 1
        xl2 = 2.0 * x1
        xr2 = x1 * 2.0
        assert xl1.real == xr1.real and xl1.dual == xr1.dual, "add reflective operator is not implemented correctly"
        assert xl2.real == xr2.real and xl2.dual == xr2.dual, "mul reflective operator is not implemented correctly"
        """test oepration between dual and dual"""
        x2 = DualNumber(5, 6)
        xl3 = x1 + x2 
        xr3 = x2 + x1
        xl4 = x1 * x2 
        xr4 = x2 * x1
        assert xl3.real == xr3.real and xl3.dual == xr3.dual, "add reflective operator is not implemented correctly"
        assert xl4.real == xr4.real and xl4.dual == xr4.dual, "mul reflective operator is not implemented correctly"

    def test_comparisons(self):
        #test whether comparison operators work for the DualNumber type
        """test __eq__"""
        x1 = DualNumber(3,2)
        x2 = DualNumber(3,2)
        x3 =3
        assert x1 == x2, "__eq__ is not working correctly"
        assert x1 == x3, "__eq__ is not working correctly"
        """test __ne__"""
        x4 = DualNumber(4,2)
        x5 = 100
        assert x1 != x4, "__ne__ is not working correctly"
        assert x1 != x5, "__ne__ is not working correctly"
        """test __lt__"""
        x6 = DualNumber(5,2)
        x7 =22.5
        x8 =0.5
        assert x1 < x6, "__lt__ is not working correctly"
        assert x1 < x7, "__lt__ is not working correctly"
        """test __le__"""
        assert x1 <= x6, "__le__ is not working correctly"
        assert x1 <= x7, "__le__ is not working correctly"
        """test __gt__"""
        assert x6 > x4, "__gt__ is not working correctly"
        assert x7 > x4, "__gt__ is not working correctly"
        assert x4 > x8, "__gt__ is not working correctly"
        """test __ge__"""
        assert x6 >= x4, "__ge__ is not working correctly"
        assert x7 >= x4, "__ge__ is not working correctly"
        assert x4 >= x8, "__gt__ is not working correctly"
