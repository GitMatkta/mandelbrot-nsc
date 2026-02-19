import numpy as np
import matplotlib.pyplot as plt
from doctest import Example
import time
import statistics

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

def benchmark ( function, max_iter = max_iter , n_runs =3) :
#""" Time func , return median of n_runs . """
    times = []
    for _ in range ( n_runs ):
        t0 = time . perf_counter ()
        result = function(max_iter)
        times.append ( time.perf_counter () - t0 )
    median_t = statistics.median ( times )
    print (f" Median : {median_t :.4f}s "f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    return median_t, result

t , M = benchmark ( mandlebrotIsolated, 100)

mandlebrotArray = mandlebrotIsolated(max_iter)

#result = mandlebrotArray
#elapsed = time.time() - start
#print(f"Execution time: {elapsed:.2f} seconds")

# plt.imshow(mandlebrotArray, extent=(x_min, x_max, y_min, y_max), cmap='twilight', origin='lower')
# plt.colorbar()
# plt.title('Mandelbrot Set')
# plt.show()
# plt.savefig('mandelbrot.png')