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




#time at 1024x1024 : 0.06 seconds

#line: kernprof -l -v mandelbrot.py
os.environ['LINE_PROFILE'] = '1'

x_min = -2
x_max = 1
y_min = -1.5
y_max = 1.5
res = 1024
max_iter = 100
power = 2
bound = 2

# x_values = np.linspace(x_min, x_max, res)
# y_values = np.linspace(y_min, y_max, res)
#@line_profiler.profile
#def mandlebrot(max_iter, x_values = x_values, y_values = y_values):
@njit
def mandlebrot(max_iter, x_values, y_values):  # accepts the arrays directly
    npArray = np.zeros((res, res), dtype=np.int32)
    for y in range(res):
        for x in range(res):
            c = x_values[x] + y_values[y] * 1j
            z = c * 0
            n = 0
            while n < max_iter and z.real * z.real + z.imag * z.imag <= bound * bound:
                z = z * z + c
                n += 1
            npArray[y, x] = n
    return npArray

def mandlebrot_typed(max_iter, dtype):
    x = np.linspace(x_min, x_max, res).astype(dtype).astype(np.float32)
    y = np.linspace(y_min, y_max, res).astype(dtype).astype(np.float32)
    return mandlebrot(max_iter, x, y)

for dtype in [np.float16, np.float32, np.float64]:
    t0 = time.perf_counter()
    mandlebrot_array = mandlebrot_typed(max_iter, dtype)
    print(f"{dtype.__name__}: {time.perf_counter() - t0:.3f} seconds")


r16 = mandlebrot_typed(max_iter, np.float16)
r32 = mandlebrot_typed(max_iter, np.float32)
r64 = mandlebrot_typed(max_iter, np.float64)

# everything below stays exactly as you had it
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, npArray, title in zip(axes, [r16, r32, r64], ['float16', 'float32', 'float64 (ref)']):
    ax.imshow(npArray, cmap='twilight')
    ax.set_title(title); ax.axis('off')
plt.savefig('precision_comparison.png', dpi=150)
plt.show()

print(f"Max diff float32 vs float64 : {np.abs(r32 - r64).max()}")
print(f"Max diff float16 vs float64 : {np.abs(r16 - r64).max()}")

#@njit
#@line_profiler.profile
#def mandlebrotpoint(c, max_iter):
    # z = 0
    # for n in range(max_iter):
    #     z = z**power + c
    #     if (abs(z) > bound):
    #         return n
    # else:
    #     return max_iter


#mandlebrotArray = mandlebrot(2)
# start = time.time()
# mandlebrotArray = mandlebrot(max_iter)
# result = mandlebrotArray
# elapsed = time.time() - start
# print(f"Execution time: {elapsed:.2f} seconds")

    # print(mandlebrot(0+0j, 100))
    # print(mandlebrot(2+2j, 100))
#print(mandlebrotArray.shape)

# def test_numba_mandelbrot_grid():
#     start_time = time.perf_counter()
#     mandelbrot_array = mandlebrot(max_iter)
#     test_time = time.perf_counter() - start_time
#     print(f'Computation took {test_time:.5f} seconds!')
#     return test_time

# num_samples = 5
# test_times = []

# for sample in range(num_samples):
#     test_time = test_numba_mandelbrot_grid()
#     test_times.append(test_time)

# numba_median_time = statistics.median(test_times)
# print(f'Median computation time: {numba_median_time:.5f} seconds!')

# plt.imshow(mandlebrotArray, extent=(x_min, x_max, y_min, y_max), cmap='twilight', origin='lower')
# plt.colorbar()
# plt.title('Mandelbrot Set')
# plt.show()
# plt.savefig('mandelbrot.png')

# print
    # TODO : Implement the algorithm