import os
import numpy as np
import matplotlib.pyplot as plt
from doctest import Example
import time
import line_profiler
from numba import njit
import statistics
from multiprocessing import Pool
from pathlib import Path


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
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    """Compute mandelbrot for rows [row_start, row_end). 
    Derives pixel coordinates from index + bounds — no arrays received as input.
    Returns a (row_end - row_start) x N int32 array."""
    out = np.zeros((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter):
    """Thin wrapper: computes the whole grid as one chunk."""
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


#grid = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter) #leftover

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers=4):
    chunk_size = max(1, N// n_workers)
    chunks, row =[], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)

if __name__ == '__main__':
    # Warm up serial first
    mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

    def test_numba_mandelbrot_grid(): #test code
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

    result = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter,n_workers=4) #actual run

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent = [x_min, x_max, y_min, y_max], cmap='inferno', origin ='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    out = Path(__file__).parent / 'mandelbrot_parallel.png'
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')
# plt.imshow(grid, extent=(x_min, x_max, y_min, y_max), cmap='twilight', origin='lower')
# plt.colorbar()
# plt.title('Mandelbrot Set')
# plt.savefig('mandelbrot.png')
# plt.show()