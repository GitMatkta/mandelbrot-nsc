import os
import numpy as np
import matplotlib.pyplot as plt
from doctest import Example
import time
import line_profiler
from numba import njit
import statistics
#Mandelbrot Set Generator
#Author : [ Me ]
#Course : Numerical Scientific Computing 2026

#def f(x):
    
    #Example function .
    #Parameters



#time at 512x512 : 1.14 seconds
#time at 1024x1024 : 4.37 seconds
#time at 2048x2048 : 17.46 seconds

#line: kernprof -l -v mandelbrot.py
os.environ['LINE_PROFILE'] = '1'

x_min = -2
x_max = 1
y_min = -1.5
y_max = 1.5
hight = 1024
width = 1024
max_iter = 100
power = 2
bound = 2

#@line_profiler.profile
@njit
def mandlebrot(max_iter):
    mandlebrotArray = []
    x_values = np.linspace(x_min, x_max, hight)
    y_values = np.linspace(y_min, y_max, width)
   #x_values, y_values = np.meshgrid(x_values, y_values)
    #c = x_values + 1j * y_values

    for y in y_values:
        row = []
        for x in x_values:
            c = complex(x, y)
            m = mandlebrotpoint(c, max_iter)
            row.append(m)
        mandlebrotArray.append(row)

    return mandlebrotArray

#@line_profiler.profile
@njit
def mandlebrotpoint(c, max_iter):
    z = 0
    for n in range(max_iter):
        z = z**power + c
        if (abs(z) > bound):
            return n
    else:
        return max_iter

#start = time.time()
#mandlebrotArray = mandlebrot(max_iter)

def test_numba_mandelbrot_grid():
    start_time = time.perf_counter()
    mandelbrot_array = mandlebrot(max_iter)
    test_time = time.perf_counter() - start_time
    print(f'Computation took {test_time:.5f} seconds!')
    return test_time

num_samples = 5
test_times = []

for sample in range(num_samples):
    test_time = test_numba_mandelbrot_grid()
    test_times.append(test_time)

numba_median_time = statistics.median(test_times)
print(f'Median computation time: {numba_median_time:.5f} seconds!')

    # print(mandlebrot(0+0j, 100))
    # print(mandlebrot(2+2j, 100))
#print(mandlebrotArray.shape)

# result = mandlebrotArray
# elapsed = time.time() - start
# print(f"Execution time: {elapsed:.2f} seconds")

# plt.imshow(mandlebrotArray, extent=(x_min, x_max, y_min, y_max), cmap='twilight', origin='lower')
# plt.colorbar()
# plt.title('Mandelbrot Set')
# plt.show()
# plt.savefig('mandelbrot.png')

# print
    # TODO : Implement the algorithm