"""
Compute and benchmark a Mandelbrot set grid.

This module generates a 2D grid of Mandelbrot iteration counts
and measures execution time over multiple runs.
"""

import os
import numpy as np
import time
import statistics

os.environ['LINE_PROFILE'] = '1'

""" Bounds of the complex plane """
x_min = -2
x_max = 1
y_min = -1.5
y_max = 1.5
hight = 1024
width = 1024
max_iter = 100
power = 2
bound = 2


def mandlebrot(max_iter):
    """
    Generate a Mandelbrot set grid.

    Parameters
    max_iter : int
        Maximum number of iterations for each point.

    Returns
    -------
    list[list[int]]
        2D list where each element represents the number of
        iterations before divergence for that complex point.
    """
    mandlebrotArray = []
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
    """
    Compute the Mandelbrot iteration count for a single complex point.

    Parameters
    ----------
    c : complex
        Complex number representing a point in the plane.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Number of iterations before divergence, or max_iter if bounded.
    """
    z = 0
    for n in range(max_iter):
        z = z**power + c
        if abs(z) > bound:
            return n
    return max_iter


def test_numba_mandelbrot_grid():
    """
    Measure execution time of Mandelbrot grid computation.

    Returns
    -------
    float
        Time taken (in seconds) to compute the Mandelbrot grid.
    """
    start_time = time.perf_counter()
    mandlebrot(max_iter)
    test_time = time.perf_counter() - start_time
    print(f'Computation took {test_time:.5f} seconds!')
    return test_time


# Run benchmark
num_samples = 5
test_times = []

for _ in range(num_samples):
    test_time = test_numba_mandelbrot_grid()
    test_times.append(test_time)

numba_median_time = statistics.median(test_times)
print(f'Median computation time: {numba_median_time:.5f} seconds!')