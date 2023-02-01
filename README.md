# AutoDiff - An Automatic Differentiation Python Package

[![.github/workflows/test.yml](https://github.com/DavidW99/AutoDiff/actions/workflows/test.yml/badge.svg)](https://github.com/DavidW99/AutoDiff/actions/workflows/test.yml)
[![.github/workflows/coverage.yml](https://github.com/DavidW99/AutoDiff/actions/workflows/coverage.yml/badge.svg)](https://github.com/DavidW99/AutoDiff/actions/workflows/coverage.yml)

This package is the final project for AC207/CS107 of 2022 at Harvard. You may use this package to do automatic differentiation, gradient descent, regression, train a neural network, and other tasks which will use derivative. 

The mathematical foundation behind this automatic differentiation package is dual number. We basically overload the elementary operations and functions with this formulation. The purpose of thei package is instructional than functional, but the AutoDiff itself is quite robust with many testing. Enjoy your usage!

## Group Members: 

Ziqing Luo, Menghang Wang, Vivian Wei, Peter Wu, Sophia Yang

## Installation
To install the package, a user shall have a Python version >= 3.7, `numpy` >= 1.20.3 and `matplotlib` >= 3.4.3 as the basic requirement. This package is Operating System independent. 

This package is released on TestPyPI [https://test.pypi.org/project/cs107-AutoDiff/] under the name "cs107-AutoDiff." The latest version is 1.0.4.

One can install the package by simply copying the following line in a Terminal:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ cs107-AutoDiff
```
which will create a "AutoDiff" folder and a "cs107_autodiff-1.0.3.dist-info" folder in the directory of the user's default Python packages.

To import the package in Python, type the following lines:

```python
from AutoDiff.AutoDiff import AutoDiff as AD
from AutoDiff.AutoDiff import exam_user_function
from AutoDiff.ElementFunction import *
from AutoDiff.DualNumber import DualNumber
from AutoDiff.feature import grad_descent
import numpy as np
```

## Use through git clone 
You can also use the package without installing it by cloning the repository from GitHub. Pasting the following line into your terminal:
  
```
git clone git@code.harvard.edu:CS107/team41.git
```
Then, you may find the there is a `usage/` folder in the repository. You can start with basic features (getting the value of and the derivative of a function) with `use_basic.py`. You can try our additional feature of doing gradient descent with `grad_descent.py`.  

## Software Organization  
There are 4 modules in this package: 
- `AutoDiff.py`: This module includes an AutoDiff class which allows the user to initialize an AutoDiff object **f** with a function $f(x_1,x_2,...,x_m):\mathbb{R}^m\mapsto\mathbb{R}^n$. The user then can use this object to calculate the value and derivative of that function on the input with given values of $[x_1,x_2,...,x_m]$ . 
   
- `DualNumber.py`: This module includes a DualNumber class, which allows the user to define a Dual Number object. 
   
- `ElementFunction.py`: This module contains functions such as **sin()**, **cos()**, **tan()**, **exp()** and **log()** that can be applied to Dual Number objects. 

- `feature.py`: This module contains the **grad_descent()** function that can perform gradient descent on a user input function and initial input value.

## Our extension: Gradient Descent

In mathematics, gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. This feature will require the user to input a scalar function $f(\vec{x}): \mathbb{R}^n \to \mathbb{R}$ with an initial guess $\vec{x_0}$, and then it will output the value of $\vec{x}$ corresponding to the local minimum of $f$. 

In addition, the user can tune this method based on their own choice of learning rate `alpha`, convergence criteria `converge_threshold`, and maximum iteration `max_iter`. If `verbose=True`, the method will print out info about $\vec{x}$ and $f(\vec{x})$ every a few hundred iterations during the gradient descent process. Please note that the output may vary if the function has multiple local minimas and the user pick different initial guesses. 

```python
x = grad_descent(function, x0, alpha=1e-5, max_iter=int(1e6), converge_threshold=1e-8, verbose=True)
```

## Documentation
Please refer to the [documentation](docs/documentation.md) here for more information, including the introduction, background and more demos. 

## Broader Impact and Inclusivity Statement

- **Broader Impact**: Automatic differentation is a far-reaching and profound tool that can be utilized in some way, shape, or form by scientists across the world. It can be used in training deep neural networks where one must continually compute the derivative of the loss function with respect to the weights for tasks like image classification or sentiment analysis, modeling differential equations to understand better fluid dynamics and different physical properties of liquids and gas we observe in the real world, and even practically in social sciences like economics where it is crucial to compute exact and precise measurements related to investment gains or losses. Everyone is able to take this software for replication or to improve upon it on their own. We stress though that this must be done in an ethical and transparent manner. Changes that ones make to the software should not substantially deter or prevent disadvantaged groups from contributing to the software in the future. For example, if one decides to adds new functions or tools to the software, it is paramount that one documents how and why they added to the existing software, and how one in the future is able to utilize these new tools. Furthermore, every change, however trivial, must be documented exactly. In order to promote a responsible and ethical environment, science empowers work to be made repeatable for anyone else to completely replicate.

- **Inclusitivity**: Our team designed `AutoDiff` to be as inclusive and accessible to the broader community as possible. Everyone who contributes to the package or interacts with it must respectful and inclusive to everyone else using it at all times. We have a zero-tolerance policy on racism, sexism, ageism, or discimination of any kind against users using this package. We welcome a diverse and well-rounded set of opinions, fostered by continual discussion and respectful constructive criticism in improving the package. We encourage users to constantly ask questions about decisions made in the package in a respectful manner. We believe anyone, regardless of background, race, or gender, can contribute to the package in a meaningful and significant way. Pull requests will be evaluated anonymously by at least two developers to lessen any kinds of bias we may have, like unconscious bias or confirmation bias. All text must be understandable by non-native English speakers. Documentation must be clear and concise. Additionally, we have a zero-tolernace policy towards ethical or disciminatory uses of the package.

## Team contribution

This is the record for the team contribution as of the completion of the project

<img src = "contribution_history.png" width=50% height= 30%>

