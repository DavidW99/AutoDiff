# All functions:
# '__add__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', 
# '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__',
# '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__mul__', 
# '__ne__', '__neg__', '__new__', '__pos__', '__pow__', '__radd__', '__reduce__', 
# '__reduce_ex__', '__repr__', '__rmul__', '__rpow__', '__rsub__', 
# '__rtruediv__', '__setattr__', '__sizeof__', '__str__', '__sub__', 
# '__subclasshook__', '__truediv__', '__weakref__', 'dual', 'real'
import pdb
import numpy as np

def check_type_convert(other):
    """
    Function to convert int/float to a DualNumber with the same real part but with zero for its dual part. If a DualNumber is inputted, then that same exact DualNumber is also outputted with no changes.
    
    Input Arguments:
    =================
    other: int/float or DualNumber

    Return Arguments:
    =================
    self: Object of the DualNumber class

    Examples
    =================
    x1 = check_type_convert(4) 
    str(x1) # 4 + 0 eps
    """
    if isinstance(other, DualNumber):
        return other
    else:
        return DualNumber(other, 0.0)

class DualNumber:
    # BEGIN CONSTRUCTOR ----------------------------------------------------
    def __init__(self, real, dual=1.0):
        """Constructor for the DualNumber class.
        
        Input Arguments:
        =================
        real: int/float
        dual: int/float
        
        Return Arguments:
        =================
        self: Object of the DualNumber class
        
        Examples
        =================
        x1 = DualNumber(4, 1.2)
        x2 = DualNumber(2.2, 6)
        x3 = DualNumber(2, 1)
        """
        if isinstance(real, (int, float)):
            self._real = real
        else:
            raise TypeError("Real part of DualNumber must be an int or float.")
        if isinstance(dual, (int, float)):
            self._dual = dual
        else:
            raise TypeError("Dual part of DualNumber must be an int or float.")
    
    # BEGIN GETTER ---------------------------------------------------------
    @property
    def real(self):
        """Function to return the private attribute of the real part of a DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        
        Return Arguments:
        =================
        The real part of a DualNumber
        
        Example
        =================
        DualNumber(3.2, 1.7).real # 3.2
        """
        return self._real
    
    # BEGIN SETTER ------------------------------------------------
    @property
    def dual(self):
        """Function to return the private attribute of the dual part of a DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        
        Return Arguments:
        =================
        The dual part of a DualNumber
        
        Example
        =================
        DualNumber(3.2, 1.7).dual # 1.7
        """
        return self._dual

    # BEGIN STR ----------------------------------------------------
    def __str__(self):
        """Returns a user-readable string output for a DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        
        Return Arguments:
        =================
        String with user-readable real part and dual part.
        
        Example
        =================
        x1 = DualNumber(3.2, 1.7)
        str(x1) # 3.2 + 1.7 eps
        """
        return f"{self._real} + {self._dual} eps"

    # BEGIN NEG -----------------------------------------------------
    def __neg__(self):
        """Returns the negation of a DualNumber by overloading the negation operator.
        
        Input Arguments:
        =================
        self: DualNumber
        
        Return Arguments:
        =================
        A DualNumber with negated attributes
        
        Example
        =================
        x1 = -DualNumber(3, 1) # DualNumber(-3, -1)
        """
        return DualNumber(-self._real, -self._dual)
                              
    # BEGIN POS -----------------------------------------------------
    def __pos__(self):
        """Returns the positive value of a DualNumber by overloading the positive operator.
        
        Input Arguments:
        =================
        self: DualNumber
        
        Return Arguments:
        =================
        A DualNumber with positive attributes
        
        Example
        =================
        x1 = +DualNumber(3, 1) # DualNumber(3, 1)
        """
        return DualNumber(self._real, self._dual)

    # BEGIN ADD -----------------------------------------------------
    def __add__(self, other):
        """Overload the addition operator to enable the addition of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of adding self and other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7)
        str(x1 + x2) # 5 + 8 eps
        """
        other = check_type_convert(other)
        return DualNumber(self._real + other._real, self._dual + other._dual)

    def __radd__(self, other): 
        """Perform same as __add__ but handle input reversal for the addition of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of adding self and other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        str(2 + x1) # 5 + 1 eps
        """
        return self.__add__(other)
    
    # BEGIN SUB -----------------------------------------------------
    def __sub__(self, other):
        """Overload the subtraction operator to enable the subtraction of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of subtracting self and other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7)
        str(x1 - x2) # 1 + - 6 eps
        """
        other = check_type_convert(other)
        return DualNumber(self._real - other._real, self._dual - other._dual)
        
    def __rsub__(self, other): 
        """Perform same as __sub__ but handle input reversal for the subtraction of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of subtracting other and self.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        str(5 - x1) # 2  + -1.0 eps
        """
        other = check_type_convert(other)
        return other.__sub__(self)
    
    # BEGIN MUL -----------------------------------------------------
    def __mul__(self, other):
        """Overload the multiplication operator to enable the multiplication of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of multiplying self and other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7)
        str(x1 * x2) # 6 + 23 eps
        """
        other = check_type_convert(other)
        return DualNumber(self._real * other._real, self._real * other._dual + self._dual * other._real)

    def __rmul__(self, other):
        """Perform same as __sub__ but handle input reversal for the multiplication of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of multiplying self and other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7)
        str(5 * x2) # 10  + 35.0 eps
        """
        return self.__mul__(other)
    
    # BEGIN TRUDIV -----------------------------------------------------
    # https://en.wikipedia.org/wiki/Dual_number
    def __truediv__(self, other):
        """Overload the division operator to enable the division of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of dividing self and other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7) 
        print(str(x1 / x2)) # 1.5  + -4.75 eps
        """
        other = check_type_convert(other)
        return DualNumber(self._real/other._real, (self._dual*other._real-self._real*other._dual)/other._real**2)
    
    def __rtruediv__(self, other):
        """Perform same as __truediv__ but handle input reversal for the division of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of multiplying self and other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7) 
        print(str(5 / x2)) # 2.5  + -8.75 eps
        """
        other = check_type_convert(other)
        return other.__truediv__(self)

    # BEGIN POW -----------------------------------------------------
    # https://math.stackexchange.com/questions/1914591/dual-number-ab-varepsilon-raised-to-a-dual-power-e-g-ab-varepsilon
    # used numpy.log()
    # define 0**0 = 1
    def __pow__(self, other):
        """Overload the exponent operator to enable the exponentation of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of computing self raised to the power of other.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7) 
        print(str(x1 ** x2)) # 9  + 75.21257418609092 eps
        """
        other = check_type_convert(other)
        if other._dual == 0:
            return DualNumber(self._real**other._real, self._real**other._real*(self._dual*other._real/self._real))
        else:
            return DualNumber(self._real**other._real, self._real**other._real*(self._dual*other._real/self._real + other._dual*np.log(self._real)))
    
    def __rpow__(self, other):
        """Perform same as __pow__ but handle input reversal for the exponentation of two objects of the DualNumber class.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        A DualNumber that is the result of computing other raised to the power of self.
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(2, 7) 
        print(str(2 ** x1)) # 8  + 5.545177444479562 eps
        """
        other = check_type_convert(other)
        return other.__pow__(self)
              
                              
    # BEGIN EQ ---------------------------------------------------------------
    def __eq__(self, other):
        """Determines whether one DualNumber equals another DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        Boolean of True if self == other, False otherwise
        
        Example
        =================
        x1 = DualNumber(4, 1)
        x2 = DualNumber(4, 1)
        x1 == x2 # True
        """
        try:
            return (self._real == other._real)
        except AttributeError:
            return (self._real == other)
                                
    # BEGIN NE -----------------------------------------------------
    def __ne__(self, other):
        """Determines whether one DualNumber is not equal to another DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        Boolean of True if self != other, False otherwise
        
        Example
        =================
        x1 = DualNumber(4, 1)
        x2 = DualNumber(2, 1)
        x1 != x2 # True
        """
        try:
            return (self._real != other._real)
        except AttributeError:
            return (self._real != other)
                              
    # BEGIN LT -----------------------------------------------------
    def __lt__(self, other):
        """Determines whether one DualNumber is less than another DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        Boolean of True if self < other, False otherwise
        
        Example
        =================
        x1 = DualNumber(4, 1)
        x2 = DualNumber(2, 1)
        x1 < x2 # False
        """
        try:
            return (self._real < other._real)
        except AttributeError:
            return (self._real < other)
                              
    # BEGIN LE -----------------------------------------------------
    def __le__(self, other):
        """Determines whether one DualNumber is less than or equal to another DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        Boolean of True if self <= other, False otherwise
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(6, 1)
        x1 <= x2 # True
        """
        try:
            return (self._real <= other._real)
        except AttributeError:
            return (self._real <= other)
                              
    # BEGIN GT -----------------------------------------------------
    def __gt__(self, other):
        """Determines whether one DualNumber is greater than another DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        Boolean of True if self > other, False otherwise
        
        Example
        =================
        x1 = DualNumber(3, 1)
        x2 = DualNumber(1, 1)
        x1 > x2 # True
        """
        try:
            return (self._real > other._real)
        except AttributeError:
            return (self._real > other)
        
    # BEGIN GE -----------------------------------------------------
    def __ge__(self, other):
        """Determines whether one DualNumber is greater than or equal to another DualNumber.
        
        Input Arguments:
        =================
        self: DualNumber
        other: DualNumber
        
        Return Arguments:
        =================
        Boolean of True if self >= other, False otherwise
        
        Example
        =================
        x1 = DualNumber(7, 1)
        x2 = DualNumber(4, 1)
        x1 >= x2 # True
        """
        try:
            return (self._real >= other._real)
        except AttributeError:
            return (self._real >= other)
    
