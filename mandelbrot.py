import numpy as np
import matplotlib.pyplot as plt
from doctest import Example


#Mandelbrot Set Generator
#Author : [ Me ]
#Course : Numerical Scientific Computing 2026

#def f(x):
    
    #Example function .
    #Parameters

mandlebrotArray = []

x_min = (-2)
x_max = (1)
y_min = (-1.5)
y_max = (1.5)
hight = 1024
width = 1024
max_iter = 100
power = 2
bound = 2


def mandlebrot(max_iter):
    x_values = np.linspace(x_min, x_max, hight)
    y_values = np.linspace(y_min, y_max, width)

    for y in y_values:
        row = []
        for x in x_values:
            c = complex(x, y)
            m = mandlebrotpoint(c, max_iter)
            row.append(m)
        mandlebrotArray.append(row)

    return mandlebrotArray


def mandlebrotpoint(c, max_iter):
    z = 0
    for n in range(max_iter):
        z = z**power + c
        if (abs(z) > bound):
            return n
    else:
        return max_iter

mandlebrotarray = mandlebrot(max_iter)

    # print(mandlebrot(0+0j, 100))
    # print(mandlebrot(2+2j, 100))
#print(mandlebrotArray.shape)

plt.imshow(mandlebrotarray, extent=(x_min, x_max, y_min, y_max), cmap='twilight', origin='lower')
plt.colorbar()
plt.title('Mandelbrot Set')
plt.show()
plt.savefig('mandelbrot.png')

# print
    # TODO : Implement the algorithm