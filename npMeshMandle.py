import os
import numpy as np
import matplotlib.pyplot as plt
from doctest import Example
import time
import statistics
import line_profiler
from numba import njit

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

#os.environ['LINE_PROFILE'] = '1' #enable for line_profiler
#@njit
#@line_profiler.profile
def mandlebrotIsolated(max_iter):
    #mandlebrotArray = np.array([])
    x_values = np.linspace(x_min, x_max, hight)
    y_values = np.linspace(y_min, y_max, width)
    X_values, Y_values = np.meshgrid(x_values, y_values)
    #print (f" Shape : {c. shape }") # (1024 , 1024)
    #print (f" Type : {c. dtype }") # complex128

    c = X_values + 1j * Y_values
    z = np.zeros_like(c)
    m = np.zeros_like(c, dtype=int)
    for n in range(max_iter):
        mask = np.abs(z) <= bound
        z[mask] = z[mask]**power + c[mask]
        m[mask] += 1
    return m


#benchmarking function made for testing purposes , not used in final code, maybe needs it's own file.

#def benchmark ( function, max_iter = max_iter , n_runs =3) :
#""" Time func , return median of n_runs . """
    times = []
    for _ in range ( n_runs ):
        t0 = time . perf_counter ()
        result = function(max_iter)
        times.append ( time.perf_counter () - t0 )
    median_t = statistics.median ( times )
    print (f" Median : {median_t :.4f}s "f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    return median_t, result

#t , M = benchmark ( mandlebrotIsolated, 100)


#old test
# start = time.time()
# mandlebrotArray = mandlebrotIsolated(max_iter)

# result = mandlebrotArray
# elapsed = time.time() - start
# print(f"Execution time: {elapsed:.2f} seconds")

def test_numba_mandelbrot_grid(): #new test
    start_time = time.perf_counter()
    mandelbrot_array = mandlebrotIsolated(max_iter)
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


#result = mandlebrotArray
#elapsed = time.time() - start
#print(f"Execution time: {elapsed:.2f} seconds")

# plt.imshow(mandlebrotArray, extent=(x_min, x_max, y_min, y_max), cmap='twilight', origin='lower')
# plt.colorbar()
# plt.title('Mandelbrot Set')
# plt.show()
# plt.savefig('mandelbrot.png')