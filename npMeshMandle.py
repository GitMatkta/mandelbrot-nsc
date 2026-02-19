import numpy as np
import matplotlib.pyplot as plt
from doctest import Example
import time

#Mandelbrot Set Generator
#Author : [ Me ]
#Course : Numerical Scientific Computing 2026

#def f(x):
    
    #Example function .
    #Parameters

start = time.time()




x_min = -2
x_max = 1
y_min = -1.5
y_max = 1.5
hight = 1024
width = 1024
max_iter = 100
power = 2
bound = 2


def mandlebrotIsolated(max_iter):
    mandlebrotArray = np.array([])
    x_values = np.linspace(x_min, x_max, hight)
    y_values = np.linspace(y_min, y_max, width)
    x_values, y_values = np.meshgrid(x_values, y_values)
    c = x_values + 1j * y_values
    print (f" Shape : {c. shape }") # (1024 , 1024)
    print (f" Type : {c. dtype }") # complex128


mandlebrotIsolated(max_iter)