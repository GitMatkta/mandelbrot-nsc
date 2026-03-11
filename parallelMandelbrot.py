import os
import numpy as np
import matplotlib.pyplot as plt
from doctest import Example
import time
import line_profiler
from numba import njit
import statistics

#parralell version



#time at 1024x1024 : 0.06 seconds

#line: kernprof -l -v mandelbrot.py
os.environ['LINE_PROFILE'] = '1'

x_min = -2
x_max = 1
y_min = -1.5
y_max = 1.5
N = 1024
max_iter = 100
power = 2
bound = 2


@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    """Compute escape iteration count for a single complex point c."""
    z_real = c_real
    z_imag = c_imag
    for n in range(max_iter):
        if z_real * z_real + z_imag * z_imag > 4.0:
            return n
        z_real, z_imag = z_real * z_real - z_imag * z_imag + c_real, 2.0 * z_real * z_imag + c_imag
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    """Compute mandelbrot for rows [row_start, row_end). 
    Derives pixel coordinates from index + bounds — no arrays received as input.
    Returns a (row_end - row_start) x N int32 array."""
    result = np.zeros((row_end - row_start, N), dtype=np.int32)
    for row in range(row_start, row_end):
        c_imag = y_min + row * (y_max - y_min) / (N - 1)
        for col in range(N):
            c_real = x_min + col * (x_max - x_min) / (N - 1)
            result[row - row_start, col] = mandelbrot_pixel(c_real, c_imag, max_iter)
    return result

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter):
    """Thin wrapper: computes the whole grid as one chunk."""
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


grid = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

def test_numba_mandelbrot_grid():
    start_time = time.perf_counter()
    mandelbrot_array = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
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


# plt.imshow(grid, extent=(x_min, x_max, y_min, y_max), cmap='twilight', origin='lower')
# plt.colorbar()
# plt.title('Mandelbrot Set')
# plt.savefig('mandelbrot.png')
# plt.show()