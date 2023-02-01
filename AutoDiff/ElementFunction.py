import numpy as np
from .DualNumber import DualNumber
# import pdb

def sin(z):
    """
    Function to calculate sin of an int/float/Dual Number. If input is a list, calculate the sin of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = sin(DualNumber(3,2)) 
    sin(x1) # sin(3) + cos(3) * 2 eps
    """
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.sin(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.sin(z_i.real), np.cos(z_i.real) * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.sin(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.sin(z.real), np.cos(z.real) * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")
            
def arcsin(z):
    """
    Function to calculate arcsin of an int/float/Dual Number. If input is a list, calculate the arcsin of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = arcsin(DualNumber(3,2)) 
    arcsin(x1) # arcsin(3) + 1/sqrt(1-9) * 2 eps
    """
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.arcsin(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.arcsin(z_i.real), 1/np.sqrt(1 - z_i.real**2) * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.arcsin(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.arcsin(z.real), 1/np.sqrt(1 - z.real**2) * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")

def cos(z):
    """
    Function to calculate cos of an int/float/Dual Number. If input is a list, calculate the cos of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = cos(DualNumber(3,2)) 
    cos(x1) # cos(3) - sin(3) * 2 eps
    """
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.cos(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.cos(z_i.real), -np.sin(z_i.real) * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.cos(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.cos(z.real), -np.sin(z.real) * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")
            
def arccos(z):
    """
    Function to calculate arccos of an int/float/Dual Number. If input is a list, calculate the arccos of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = arccos(DualNumber(3,2)) 
    arccos(x1) # arccos(3) - 1/sqrt(1-9) * 2 eps
    """
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.arccos(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.arccos(z_i.real), -1/np.sqrt(1 - z_i.real**2) * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.arccos(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.arccos(z.real), -1/np.sqrt(1 - z.real**2) * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")


def tan(z):
    """
    Function to calculate tan of an int/float/Dual Number. If input is a list, calculate the tan of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = tan(DualNumber(3,2)) 
    tan(x1) # tan(3) + 1/cos(3)^2 * 2 eps
    """
    ### add ValueError for tan(np.pi/2)
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                if z_i%(np.pi) == np.pi/2:
                    raise ValueError("The input for tan() cannot be pi/2 + n*pi")
                else:
                    fz.append(np.tan(z_i))
            elif isinstance(z_i, DualNumber):
                if z_i.real%(np.pi) == np.pi/2:
                    raise ValueError("The input for tan() cannot be pi/2 + n*pi")
                else:
                    fz.append(DualNumber(np.tan(z_i.real), z_i.dual / np.cos(z_i.real) ** 2))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            if z%(np.pi) == np.pi/2:
                raise ValueError("The input for tan() cannot be pi/2 + n*pi")
            else:
                return np.tan(z)
        elif isinstance(z, DualNumber):
            if z.real%(np.pi) == np.pi/2:
                raise ValueError("The input for tan() cannot be pi/2 + n*pi")
            else:
                return DualNumber(np.tan(z.real), z.dual / np.cos(z.real) ** 2)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")

def arctan(z):
    """
    Function to calculate arctan of an int/float/Dual Number. If input is a list, calculate the arctan of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = arcsin(DualNumber(3,2)) 
    arctan(x1) # arctan(3) + 1/10 * 2 eps
    """
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.arctan(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.arctan(z_i.real), z_i.dual/(1+z_i.real**2)))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.arctan(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.arctan(z.real), z.dual/(1+z.real**2))
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")
            
def exp(z):
    """
    Function to calculate exponential of an int/float/Dual Number. If input is a list, calculate the exponential of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = exp(DualNumber(3,2)) 
    exp(x1) # exp(3) + exp(3) * 2 eps
    """
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.exp(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.exp(z_i.real), np.exp(z_i.real) * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.exp(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.exp(z.real), np.exp(z.real) * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")

def log(z):
    """
    Function to calculate logarithm of an int/float/Dual Number. If input is a list, calculate the logarithm of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = log(DualNumber(3,2)) 
    log(x1) # log(3) + 1/3 * 2 eps
    """
    if isinstance(z, (list, np.ndarray)):
        fz = []
        for z_i in z:
            if isinstance(z_i, DualNumber):
                ## float.real exists, but float.dual does not
                if z_i.real <= 0:
                    raise ValueError("The function input vector contains Dual Number whose real part is non-positive")
                else:
                    fz.append(DualNumber(np.log(z_i.real), z_i.dual / z_i.real))
                    #if isinstance(z_i, (int, float)):
                    #    fz.append(np.log(z_i))
            elif isinstance(z_i,(int, float, np.number)):
                if z_i <= 0:
                    raise ValueError('The function input vector contains a non-positive number')
                else:
                    fz.append(np.log(z_i))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        # if isinstance(z, np.ndarray):
        #     return np.array(fz)
        return np.array(fz)
    else:
        if isinstance(z, DualNumber):
            if z.real <= 0:
                raise ValueError("The function input is a Dual Number whose real part is non-positive")
            else:
                return DualNumber(np.log(z.real), z.dual / z.real)
        elif isinstance(z, (int,float)):
                if z <= 0:
                    raise ValueError('The function input is non-positive')
                else:
                    return np.log(z)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")

#David's Version
# def log(z):
#     if isinstance(z, list):
#         fz = []
#         for z_i in z:
#             if isinstance(z_i, (int, float, DualNumber)):
#                 ## float.real exists, but float.dual does not
#                 if z_i.real <= 0:
#                     raise ValueError("One of the function input is non-positive")
#                 else:
#                     if isinstance(z_i, (int, float)):
#                         fz.append(np.log(z_i))
#                     else:
#                         fz.append(DualNumber(np.log(z_i.real), z_i.dual / z_i.real))
#             else:
#                 raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
#         return fz
#     else:
#         if isinstance(z, (int, float, DualNumber)):
#             if z.real <= 0:
#                 raise ValueError("The function input is non-positive")
#             else:
#                 if isinstance(z, (int, float)):
#                     return np.log(z)
#                 else:
#                     return DualNumber(np.log(z.real), z.dual / z.real)
#         else:
#             raise TypeError("The function input is neither a scalar nor DualNumber")

# def pow(z, n):
#     """
#     Function to calculate the power of an int/float/Dual Number. If input is a list, calculate the power of each element. 

#     Input Arguments:
#     =================
#     z: int/float or DualNumber or a list of int/float/DualNumber

#     Return Arguments:
#     =================
#     int/float or DualNumber or a list of int/float/DualNumber

#     Examples
#     =================
#     x1 = arcsin(DualNumber(3,2)) 
#     arcsin(x1) # arcsin(3) + 1/sqrt(1-9) * 2 eps
#     """
#     if not isinstance(z, (int, float, DualNumber)):
#         raise TypeError("The function input is neither a scalar nor DualNumber")
#     if not isinstance(n, (int, float)):
#         raise TypeError("The exponent is not a scalar")
#     #DualNumber(z.real ** n, n * z.real ** (n - 1) * z.dual)
#     if isinstance(z, list):
#         fz = []
#         for z_i in z:
#             if isinstance(z_i, (int, float)):
#                 fz.append(np.pow(z_i))
#             elif isinstance(z_i, DualNumber):
#                 fz.append(DualNumber(z.real ** n, n * z.real ** (n - 1) * z.dual))
#             else:
#                 raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
#         return fz
#     else:
#         if isinstance(z, (int, float)):
#             return np.pow(z)
#         elif isinstance(z, DualNumber):
#             return DualNumber(z.real ** n, n * z.real ** (n - 1) * z.dual)
#         else:
#             raise TypeError("The function input is neither a scalar nor DualNumber")

def sqrt(z):
    """
    Function to calculate square root of an int/float/Dual Number. If input is a list, calculate the square root of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = sqrt(DualNumber(3,2)) 
    sqrt(x1) # sqrt(3) + 1/2 * 3^(-1/2) * 2 eps = sqrt(3) + 1/sqrt(3) eps
    """
 #   if not isinstance(z, (int, float, DualNumber)):
  #      raise TypeError("The function input is neither a scalar nor DualNumber")
   # return DualNumber(z.real ** 0.5, 0.5 * z.real ** (-0.5) * z.dual)
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, DualNumber):
                ## float.real exists, but float.dual does not
                if z_i.real <= 0:
                    raise ValueError("The function input vector contains Dual Number whose real part is non-positive")
                else:
                    fz.append(DualNumber(z_i.real ** 0.5, 0.5 * z_i.real ** (-0.5) * z_i.dual))
                    #if isinstance(z_i, (int, float)):
                    #    fz.append(np.log(z_i))
            elif isinstance(z_i,(int,float)):
                if z_i <= 0:
                    raise ValueError('The function input vector contains a non-positive number')
                else:
                    fz.append(np.sqrt(z_i))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, DualNumber):
            if z.real <= 0:
                raise ValueError("The function input is a Dual Number whose real part is non-positive")
            else:
                return DualNumber(z.real ** 0.5, 0.5 * z.real ** (-0.5) * z.dual)
        elif isinstance(z, (int,float)):
                if z <= 0:
                    raise ValueError('The function input is non-positive')
                else:
                    return np.sqrt(z)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")


def logBase(b,z):
    """
    Function to calculate logarithm with base b of an int/float/Dual Number. If input is a list, calculate the logarithm with base b of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = logBase(2,DualNumber(3,2)) 
    logBase(x1) # log2(3) + 1/(3 * log(2)) * 2 eps
    """
    assert b > 0 and isinstance(b,(int,float))
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, DualNumber):
                ## float.real exists, but float.dual does not
                if z_i.real <= 0:
                    raise ValueError("The function input vector contains Dual Number whose real part is non-positive")
                else:
                    fz.append(DualNumber(np.log(z_i.real)/np.log(b), z_i.dual / (z_i.real* np.log(b))))
                    #if isinstance(z_i, (int, float)):
                    #    fz.append(np.log(z_i))
            elif isinstance(z_i,(int,float)):
                if z_i <= 0:
                    raise ValueError('The function input vector contains a non-positive number')
                else:
                    fz.append(np.log(z_i.real)/np.log(b))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, DualNumber):
            if z.real <= 0:
                raise ValueError("The function input is a Dual Number whose real part is non-positive")
            else:
                return DualNumber(np.log(z.real)/np.log(b), z.dual / (z.real* np.log(b)))
        elif isinstance(z, (int,float)):
                if z <= 0:
                    raise ValueError('The function input is non-positive')
                else:
                    return np.log(z.real)/np.log(b)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")

def log10(z):
    """
    Function to calculate logarithm with base 10 of an int/float/Dual Number. If input is a list, calculate the logarithm with base 10 of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = log10(DualNumber(3,2)) 
    log10(x1) # log10(3) + 1/(3 * ln(3)) eps
    """

    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, DualNumber):
                ## float.real exists, but float.dual does not
                if z_i.real <= 0:
                    raise ValueError("The function input vector contains Dual Number whose real part is non-positive")
                else:
                    fz.append(DualNumber(np.log(z_i.real)/np.log(10), z_i.dual / (z_i.real* np.log(10))))
                    #if isinstance(z_i, (int, float)):
                    #    fz.append(np.log(z_i))
            elif isinstance(z_i,(int,float)):
                if z_i <= 0:
                    raise ValueError('The function input vector contains a non-positive number')
                else:
                    fz.append(np.log(z_i.real)/np.log(10))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, DualNumber):
            if z.real <= 0:
                raise ValueError("The function input is a Dual Number whose real part is non-positive")
            else:
                return DualNumber(np.log(z.real)/np.log(10), z.dual / (z.real* np.log(10)))
        elif isinstance(z, (int,float)):
                if z <= 0:
                    raise ValueError('The function input is non-positive')
                else:
                    return np.log(z.real)/np.log(10)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")
            
def sinh(z):
    """
    Function to calculate sinh of an int/float/Dual Number. If input is a list, calculate the sinh of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = sinh(DualNumber(3,2)) 
    sinh(x1) # sinh(3) + cosh(3) * 2 eps
    """

    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.sinh(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.sinh(z_i.real), np.cosh(z_i.real) * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.sinh(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.sinh(z.real), np.cosh(z.real) * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")

def cosh(z):
    """
    Function to calculate cosh of an int/float/Dual Number. If input is a list, calculate the cosh of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = cosh(DualNumber(3,2)) 
    cosh(x1) # cosh(3) + sinh(3) * 2 eps
    """

    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(np.cosh(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.cosh(z_i.real), np.sinh(z_i.real) * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return np.cosh(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.cosh(z.real), np.sinh(z.real) * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")


def tanh(z):
    """
    Function to calculate tanh of an int/float/Dual Number. If input is a list, calculate the tanh of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = tanh(DualNumber(3,2)) 
    tanh(x1) # tanh(3) + (cosh(3)^2 - sinh(3)^2)/cosh(3)^2 * 2 eps
    """
    if isinstance(z, (list, np.ndarray)):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float, np.number)):
                fz.append(np.tanh(z_i))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(np.tanh(z_i.real), (np.cosh(z_i.real)**2 - np.sinh(z_i.real)**2)/ np.cosh(z_i.real)**2 * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        # if isinstance(z, np.ndarray):
        #     return np.array(fz)
        return np.array(fz)
    else:
        if isinstance(z, (int, float)):
            return np.tanh(z)
        elif isinstance(z, DualNumber):
            return DualNumber(np.tanh(z.real), (np.cosh(z.real)**2 - np.sinh(z.real)**2)/np.cosh(z.real)**2 * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")

def logistic(z):
    """
    Function to calculate logistic of an int/float/Dual Number. If input is a list, calculate the logistic of each element. 

    Input Arguments:
    =================
    z: int/float or DualNumber or a list of int/float/DualNumber

    Return Arguments:
    =================
    int/float or DualNumber or a list of int/float/DualNumber

    Examples
    =================
    x1 = logistic(DualNumber(3,2)) 
    logistic(x1) # 1/(1+exp(-3)) + exp(-3)/(1+exp(-3))**2 * 2 eps
    """
    if isinstance(z, list):
        fz = []
        for z_i in z:
            if isinstance(z_i, (int, float)):
                fz.append(1/(1+np.exp(-z_i)))
            elif isinstance(z_i, DualNumber):
                fz.append(DualNumber(1/(1+np.exp(-z_i.real)), np.exp(-z_i.real)/(1+np.exp(-z_i.real))**2 * z_i.dual))
            else:
                raise TypeError("The function input vector contains a non-scalar or non-DualNumber value")
        return fz
    else:
        if isinstance(z, (int, float)):
            return 1/(1+np.exp(-z))
        elif isinstance(z, DualNumber):
            return DualNumber(1/(1+np.exp(-z.real)), np.exp(-z.real)/(1+np.exp(-z.real))**2 * z.dual)
        else:
            raise TypeError("The function input is neither a scalar nor DualNumber")
